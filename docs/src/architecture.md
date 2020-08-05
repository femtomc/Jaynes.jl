## Implementation

As we started to see in the previous section, Jaynes is organized around a central `IRTools` _dynamo_

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

# Whitelist includes vectorized calls.
whitelist = [:rand, 
             :learnable, 
             :markov, 
             :plate, 
             :cond, 
             :_apply_iterate,
             # Foreign model interfaces
             :foreign]

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

# Fix for _apply_iterate.
function f_push!(arr::Array, t::Tuple{}) end
f_push!(arr::Array, t::Array) = append!(arr, t)
f_push!(arr::Array, t::Tuple) = append!(arr, t)
f_push!(arr, t) = push!(arr, t)
function flatten(t::Tuple)
    arr = Any[]
    for sub in t
        f_push!(arr, sub)
    end
    return arr
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
    if has_query(ctx.select, addr)
        s = get_query(ctx.select, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, ChoiceSite(score, s))
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, ChoiceSite(logpdf(d, s), s))
    end
    return s
end

```

so this context records the random choice, as well as performs some bookkeeping which we use for inference. Each of the other contexts define unique interception dispatch to implement functionality required for inference over probabilistic program traces. [These can be found here.](https://github.com/femtomc/Jaynes.jl/tree/master/src/contexts)

## Record sites and traces

The conceptual entities of the Jaynes tracing system are _record sites_ and _traces_.

```julia
abstract type RecordSite end
abstract type Trace end
```
which, respectively, represent sites at which randomness occurs (and is traced) and the trace itself. Let's examine one type of site, a `ChoiceSite`:

```julia
struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end
```

A `ChoiceSite` is just a record of a random selection, along with the log probability of the selection with respect to the user-specified distribution at that site. These are created by calls of the form `rand(addr::Address, d::Distribution)` where `Distribution` is the type from the `Distributions` library.

The structured representation of recorded randomness in a program execution is a `Trace`:

```julia
struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    params::Dict{Address, LearnableSite}
end
```

which has separate fields for _call sites_ and _choice sites_:

Here, I'm showing `HierarchicalTrace` which is used in black-box calls as the default trace. Here's `VectorizedTrace` which is activated by special language calls (for now, `markov` and `plate`, likely more in the future):

```julia
struct VectorizedTrace{C <: RecordSite} <: Trace
    subrecords::Vector{C}
    params::Dict{Address, LearnableSite}
end
```

This trace explicitly represents certain dependency information in the set of calls specified by the language calls - e.g. `markov` specifies a Markovian dependency from one call to the next and `plate` specifies IID calls.

We just encountered `ChoiceSite` above in `GenerateContext` - let's look at an example `CallSite`:

```julia
struct HierarchicalCallSite{J, K} <: CallSite
    trace::HierarchicalTrace
    score::Float64
    fn::Function
    args::J
    ret::K
end
```

This call site is how we represent black box function calls which the user has indicated need to be traced. Other call sites present unique functionality, which (when traced) provide the contexts used for inference with additional information which can speed up certain operations.

Generically, these entities are all that are required to construct a set of inference APIs over program traces and the choice maps represented in those traces. Other advanced functionality (like specialized call sites) are variations on these themes.
