@def title = "Implementation architecture"

Jaynes is organized around a central [IRTools](https://github.com/FluxML/IRTools.jl) [dynamo](https://fluxml.ai/IRTools.jl/latest/dynamo/):

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. The "wrapping transform" implemented through `recur!` is customized, to make the tracer lightweight and to prevent some type stability issues from calls to `Base` Julia.

```julia
unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

whitelist = [
             # Base.
             :rand, :_apply_iterate, :collect,

             # Specialized call sites.
             :markov, :plate, :cond, 

             # Interactions with the context.
             :learnable, :fillable, :factor,

             # Foreign model interfaces.
             :foreign, :deep
            ]

# Fix for specialized tracing.
function recur!(ir, to = self)
    for (x, st) in ir
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || continue
            ir[x] = Expr(:call, to, st.expr.args...)
        end
    end
    return ir
end
```

so we see that the tracer is only allowed to look at certain calls, and uses a few fixes for some common issues. This drastically improves the performance over a "heavyweight" tracer which looks at everything. For the use case of probabilistic programming [implemented in this style](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf), it's perfectly acceptable.

There are a number of inheritors for `ExecutionContext`

```
GenerateContext
SimulateContext
ProposalContext
UpdateContext
RegenerateContext
ScoreContext
ChoiceBackpropagateContext
ParameterBackpropagateContext
```

each of which has a set of special dispatch definition which allows the dynamo to dispatch on `rand` calls with user-provided addressing. As an example, here's the interception dispatch inside the `GenerateContext` (which we just examined in the last section):

```julia
@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    if has_value(ctx.target, addr)
        s = getindex(ctx.target, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, score, s)
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, logpdf(d, s), s)
    end
    return s
end
```

so this context records the random choice, as well as performs some bookkeeping with the `logpdf` which we will use for inference programming. Each of the other contexts define unique interception dispatch to implement functionality required for inference over probabilistic program traces. [These can be found here.](https://github.com/femtomc/Jaynes.jl/tree/master/src/contexts)

## Sugar

The programmer is not expected to interact with these contexts directly. Instead, the programmer can utilize a set of high-level function calls which construct the contexts, run a function call with some arguments in the context, and return useful information (usually, a return value, a bundled record of the call in a `CallSite` instance, and some other probabilistic metadata). These high-level calls match the same high-level calls in `Gen.jl`:

* `simulate`
* `propose`
* `generate`
* `update`
* `regenerate`
* `score`

(roughly, `Gen.jl` may change their interfaces, these may also change here - but the ideas behind these interfaces will remain the same)

You use these high-level interface calls on your models e.g.

```julia
ret, cl = simulate(some_model, args...)
```

which takes care of constructing a `SimulateContext`, executing your model with `args...` in that context, and bundling up the return and a record of that call for you.

## Inference

If you examine the inference library in any detail, you'll notice that the algorithms are implemented using the interfaces above. The combined set of interfaces and contexts form a convenient set of tools for constructing inference algorithms. [As mentioned in the about page](/), this set of interfaces has been slowly crafted through a lineage of work starting with [Venture](http://probcomp.csail.mit.edu/software/venture/) and continuing in [Gen.jl](https://github.com/probcomp/Gen.jl).

Let's examine the implementation of _importance sampling_ from a proposal model.

```julia
function importance_sampling(observations::K,
                             num_samples::Int,
                             model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple) where K <: AddressMap
    calls = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    Threads.@threads for i in 1 : num_samples
        ret, pmap, pw = propose(proposal, proposal_args...)
        overlapped = Jaynes.merge!(pmap, observations)
        overlapped && error("(importance_sampling, merge!): proposal produced a selection which overlapped with observations.")
        _, calls[i], gw = generate(pmap, model, args...)
        lws[i] = gw - pw
    end
    ltw = lse(lws)
    lnw = lws .- ltw
    return Particles(calls, lws, 0.0), lnw
end
```

Importance sampling works as follows: you have a distribution $P(x)$ which is hard to sample from, so you provide a $Q(x)$ which (ideally, the first condition is not required) satisfies two conditions.

1. It's easy to sample from $Q(x)$.
2. $Q(x)$ is _absolutely continuous_ with respect to $P(x)$.

If this is true, you can compute expectations with respect to $P(x)$ by sampling from $Q(x)$ and then correcting for the fact that you're not sampling from $P(x)$.

In the code, here you sample from $P(x)$ with the call to `propose` - this produces a proposal `CallSite` (here, `pmap`) and the score of the proposal sample with respect to its own prior. Then, we merge the `pmap` into the set of observations using `merge!` - this simultaneously produces a new `AddressMap` (which will constrain addresses in any call) and checks if there is overlap. **If there is overlap, it is a support error**. Finally, we call `generate` with the new `AddressMap` and compute the importance weight `gw - pw` to store in the collection of log weights.

Now, it becomes simple to apply importance sampling inference - because it has been built from the interface calls to execution contexts.

```julia
ps, lnw = importance_sampling(obs, n_samples, 
                              some_model, (args..., ),
                              proposal, (data, ))
```
