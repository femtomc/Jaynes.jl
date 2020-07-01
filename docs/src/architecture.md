This is a library which implements probabilistic programming by intercepting calls to `rand` and interpreting them according to a user-provided context. The interception is automatic through the execution of Julia code, as the interception is provided by compiler injection into an intermediate representation of code (lowered code) using a package called [Cassette](https://github.com/jrevels/Cassette.jl).

Cassette is a very powerful package, but it's also very subtle and easy to cause deep issues in the compilation pipeline. Here, I'm not doing anything too crazy with, say, composition of contexts or compiler pass injection (yet). The basic idea is that you may have some code

```julia
function foo(x::Float64)
    y = rand(:y, Normal, (x, 1.0))
    return y
end
```

which you want to interpret in a probabilistic programming context. The lowered code looks like this:

```julia
1 ─ %1 = Core.tuple(x, 1.0)
│        y = Main.rand(:y, Main.Normal, %1)
└──      return y
```

After we intercept, the code looks like this:

```julia
1 ─      #self# = Core.getfield(##overdub_arguments#254, 1)
│        x = Core.getfield(##overdub_arguments#254, 2)
│        Cassette.prehook(##overdub_context#253, Core.tuple, x, 1.0)
│   %4 = Cassette.overdub(##overdub_context#253, Core.tuple, x, 1.0)
│        Cassette.posthook(##overdub_context#253, %4, Core.tuple, x, 1.0)
│   %6 = %4
│        Cassette.prehook(##overdub_context#253, Main.rand, :y, Main.Normal, %6)
│   %8 = Cassette.overdub(##overdub_context#253, Main.rand, :y, Main.Normal, %6)
│        Cassette.posthook(##overdub_context#253, %8, Main.rand, :y, Main.Normal, %6)
│        y = %8
│   @ REPL[1]:3 within `foo'
└──      return y
```

notice that every method invocation has been wrapped in a special function (either `prehook`, `overdub`, or `posthook`) which accepts a special structure as first argument (a _context_). For this conversation, we won't use the special `prehook` or `posthook` points of access...

```julia
1 ─      #self# = Core.getfield(##overdub_arguments#254, 1)
│        x = Core.getfield(##overdub_arguments#254, 2)
│   %4 = Cassette.overdub(##overdub_context#253, Core.tuple, x, 1.0)
│   %6 = %4
│   %8 = Cassette.overdub(##overdub_context#253, Main.rand, :y, Main.Normal, %6)
│        y = %8
│   @ REPL[1]:3 within `foo'
└──      return y
```

so we'll just discuss `overdub`. Now, a structured form for the `overdub` context allows us to record probabilistic statements to a trace. What is a _context_?

```julia
Context{N<:Cassette.AbstractContextName,
        M<:Any,
        P<:Cassette.AbstractPass,
        T<:Union{Nothing,Cassette.Tag},
        B<:Union{Nothing,Cassette.BindingMetaDictCache},
        H<:Union{Nothing,Cassette.DisableHooks}}
```

where `M` is _metadata_. We use a structured trace as metadata

```julia
mutable struct UnconstrainedGenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    UnconstrainedGenerateMeta(tr::Trace) = new(tr, Address[])
end
```

and then, with `overdub`, we intercept `rand` calls and store the correct location, value, and score in the trace inside the meta.

```julia
function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: UnconstrainedGenerateMeta, 
                                       T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    haskey(ctx.metadata.tr.chm, addr) && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    return sample
end
```

We also keep a stack around to handle hierarchical addressing inside function calls. The stack is essentially a lightweight call stack which tracks where we are while tracing. This lets us get the addressing correct, without doing too much work.

Different forms of metadata structure allow us to implement sampling-based inference algorithms efficiently. A `ProposalMeta` comes with its own `overdub` dispatch which minimizes calls during a proposal sampling routine:

```julia
@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: ProposalMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    ctx.metadata.tr.score += score
    push!(ctx.metadata.visited, addr)
    return sample

end
```

To express inference algorithms, we can mix `overdub` dispatch on contexts with certain subtypes of `Meta`. Contexts are re-used during iterative algorithms - the dominant form of allocation is new-ing up blank `Trace` instances for tracing.

```julia
function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple,
                             observations::Dict{Address, T},
                             num_samples::Int) where T
    trs = Vector{Trace}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    prop_ctx = disablehooks(TraceCtx(metadata = ProposalMeta(Trace())))
    model_ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), observations)))
    for i in 1:num_samples
        # Propose.
        if isempty(proposal_args)
            Cassette.overdub(prop_ctx, proposal)
        else
            Cassette.overdub(prop_ctx, proposal, proposal_args...)
        end

        # Merge proposals and observations.
        prop_score = prop_ctx.metadata.tr.score
        prop_chm = prop_ctx.metadata.tr.chm
        constraints = merge(observations, prop_chm)
        model_ctx.metadata.constraints = constraints

        # Generate.
        if isempty(args)
            res = Cassette.overdub(model_ctx, model)
        else
            res = Cassette.overdub(model_ctx, model, args...)
        end

        # Track score.
        model_ctx.metadata.tr.func = model
        model_ctx.metadata.tr.args = args
        model_ctx.metadata.tr.retval = res
        lws[i] = model_ctx.metadata.tr.score - prop_score
        trs[i] = model_ctx.metadata.tr

        # Reset.
        reset_keep_constraints!(model_ctx.metadata)
        reset_keep_constraints!(prop_ctx.metadata)
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
```
