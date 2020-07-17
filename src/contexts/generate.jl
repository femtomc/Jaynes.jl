mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection, P <: Parameters} <: ExecutionContext
    tr::T
    select::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
    GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K, NoParameters}(tr, select, 0.0, 0.0, Visitor(), Parameters())
    GenerateContext(tr::T, select::K, params::P) where {T <: Trace, K <: ConstrainedSelection, P <: Parameters} = new{T, K, P}(tr, select, 0.0, 0.0, Visitor(), params)
end
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)

# ------------ Convenience ------------ #

function generate(sel::L, fn::Function, args...; params = Parameters()) where L <: ConstrainedSelection
    ctx = Generate(sel, params)
    ret = ctx(fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(sel::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: ConstrainedSelection, K}
    ctx = Generate(sel, params)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_choice(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: ConstrainedSelection
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel, params)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: ConstrainedSelection
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel, params)
    ret = ctx(fn, addr, call, args)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: ConstrainedSelection, K}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel, params)
    ret = ctx(fn, addr, d, len)
    return ret, get_call(ctx.tr, addr), ctx.weight
end

# ------------ includes ------------ #

include("hierarchical/generate.jl")
include("plate/generate.jl")
include("markov/generate.jl")
include("cond/generate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: Trace, K <: ConstrainedSelection, P <: Parameters} <: ExecutionContext
     tr::T
     select::K
     weight::Float64
     score::Float64
     visited::Visitor
     params::P
end
```
`GenerateContext` is used to generate traces, as well as record and accumulate likelihood weights given observations at addressed randomness.

Inner constructors:
```julia
GenerateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, 0.0, Visitor(), Parameters())
GenerateContext(tr::T, select::K, params::P) where {T <: Trace, K <: ConstrainedSelection, P <: Parameters} = new{T, K, P}(tr, select, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(select::ConstrainedSelection) = GenerateContext(Trace(), select)
Generate(select::ConstrainedSelection, params) = GenerateContext(Trace(), select, params)
Generate(tr::Trace, select::ConstrainedSelection) = GenerateContext(tr, select)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(sel::L, fn::Function, args...; params = Parameters()) where L <: ConstrainedSelection
ret, cs, w = generate(sel::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: ConstrainedSelection, K}
ret, v_cl, w = generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: ConstrainedSelection
ret, v_cl, w = generate(sel::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: ConstrainedSelection
ret, v_cl, w = generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: ConstrainedSelection, K}
```
The convenience `generate` function provides an easy API to the `Generate` context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `sel`.
""", generate)

