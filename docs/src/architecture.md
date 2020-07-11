## Implementation

Jaynes is organized around a central `IRTools` _dynamo_

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. There are a number of such inheritors

```
GenerateContext
ProposalContext
UpdateContext
RegenerateContext
ScoreContext
BackpropagateContext
```

each of which has a special dispatch definition which allows the dynamo to dispatch on `rand` calls with addressing. As an example, here's the interception dispatch inside the `GenerateContext`

```julia
@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    if has_query(ctx.select, addr)
        s = get_query(ctx.select, addr)
        score = logpdf(d, s)
        add_choice!(ctx.tr, addr, ChoiceSite(score, s))
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx.tr, addr, ChoiceSite(logpdf(d, s), s))
    end
    visit!(ctx.visited, addr)
    return s
end

```

so this context records the random choice, as well as performs some bookkeeping which we use for inference. Each of the other contexts define unique interception dispatch to implement functionality required for inference over probabilistic program traces. 

## Traces

The structured representation of program execution is a `Trace`

```julia
abstract type Trace end
mutable struct HierarchicalTrace <: Trace
    calls::Dict{Address, CallSite}
    choices::Dict{Address, ChoiceSite}
    params::Dict{Address, LearnableSite}
    score::Float64
    function HierarchicalTrace()
        new(Dict{Address, CallSite}(), 
            Dict{Address, ChoiceSite}(),
            Dict{Address, LearnableSite}(),
            0.0)
    end
end
```

Here, I'm showing `HierarchicalTrace` which is the generic (and currently, only) trace type. We just encountered `ChoiceSite` above - let's look at an example `CallSite`

```julia
mutable struct BlackBoxCallSite{T <: Trace, J, K} <: CallSite
    trace::T
    fn::Function
    args::J
    ret::K
end
```

This call site is how we represent black box function calls which the user has indicated need to be traced. Other call sites present unique functionality, which (when traced) provide the contexts used for inference with additional information which can speed up certain operations.
