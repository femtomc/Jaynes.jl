Jaynes is a simple implementation of _effect-oriented programming_ for probabilistic programming. It closely follows the design of [Gen](https://www.gen.dev/) which also uses the notion of stateful execution contexts to produce the interfaces required for inference. Jaynes is organized around a central [IRTools](https://github.com/FluxML/IRTools.jl) _dynamo

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. There are a number of such inheritors

```
UnconstrainedGenerateContext
ConstrainedGenerateContext
ProposalContext
UpdateContext
RegenerateContext
ScoreContext
```

each of which has a special dispatch definition which allows the dynamo to dispatch on `rand` calls with addressing e.g.

```julia
@inline function (ctx::UnconstrainedGenerateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     d::Distribution{K}) where {T <: Address, K}
    s = rand(d)
    ctx.tr.chm[addr] = ChoiceSite(logpdf(d, s), s)
    return s
end
```

for `UnconstrainedGenerateContext`. Each of the other contexts define a particular functionality required for inference over probabilistic program traces. 

The structured representation of traces is also an `ExecutionContext`

```julia
abstract type Trace <: ExecutionContext end

mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    function HierarchicalTrace()
        new(Dict{Address, RecordSite}(), 0.0)
    end
end
```

so usage for unconstrained generation is simple

```julia
using Jaynes
geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)
tr = Trace()
tr(geo, 5.0)
display(tr, show_value = true)
```

will produce
