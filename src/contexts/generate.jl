mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: AddressMap} <: ExecutionContext
    tr::T
    constraints::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
    GenerateContext(tr::T, select::K) where {T <: AddressMap, K <: AddressMap} = new{T, K, Empty}(tr, select, 0.0, 0.0, Visitor(), Empty())
    GenerateContext(tr::T, select::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: AddressMap} = new{T, K, P}(tr, select, 0.0, 0.0, Visitor(), params)
end
Generate(select::AddressMap) = GenerateContext(Trace(), select, Parameters())
Generate(select::AddressMap, params) = GenerateContext(Trace(), select, params)
Generate(tr::AddressMap, select::AddressMap) = GenerateContext(tr, select)

# ------------ Convenience ------------ #

function generate(sel::L, fn::Function, args...) where L <: AddressMap
    ctx = Generate(sel)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(sel::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Generate(sel, params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(sel::L, fn::typeof(rand), d::Distribution{K}) where {L <: AddressMap, K}
    ctx = Generate(sel)
    addr = gensym()
    ret = ctx(fn, addr, d)
    return ret, get_top(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(sel::L, params, fn::typeof(markov), call::Function, len::Int, args...) where L <: AddressMap
    addr = gensym()
    v_sel = selection(addr => sel)
    v_ps = learnables(addr => params)
    ctx = Generate(v_sel, v_ps)
    ret = ctx(fn, addr, call, len, args...)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(sel::L, params, fn::typeof(plate), call::Function, args::Vector) where L <: AddressMap
    addr = gensym()
    v_sel = selection(addr => sel)
    v_ps = learnables(addr => params)
    ctx = Generate(v_sel, v_ps)
    ret = ctx(fn, addr, call, args)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

function generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int) where {L <: AddressMap, K}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Generate(v_sel)
    ret = ctx(fn, addr, d, len)
    return ret, get_sub(ctx.tr, addr), ctx.weight
end

# ------------ includes ------------ #

include("dynamic/generate.jl")
include("plate/generate.jl")
include("markov/generate.jl")
include("conditional/generate.jl")
include("factor/generate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: Parameters} <: ExecutionContext
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
GenerateContext(tr::T, select::K) where {T <: AddressMap, K <: AddressMap} = new{T, K}(tr, select, 0.0, Visitor(), Parameters())
GenerateContext(tr::T, select::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: Parameters} = new{T, K, P}(tr, select, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(select::AddressMap) = GenerateContext(AddressMap(), select)
Generate(select::AddressMap, params) = GenerateContext(AddressMap(), select, params)
Generate(tr::AddressMap, select::AddressMap) = GenerateContext(tr, select)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(sel::L, fn::Function, args...; params = Parameters()) where L <: AddressMap
ret, cs, w = generate(sel::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: AddressMap, K}
ret, v_cl, w = generate(sel::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(sel::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: AddressMap, K}
```

`generate` provides an API to the `GenerateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `sel`.
""", generate)

