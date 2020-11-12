# ------------ Staging ------------ #

@dynamo function (gx::GenerateContext{J})(a...) where J
    ir = IR(a...)
    ir == nothing && return
    ir = staged_pipeline(ir, GenerateContext{J})
    ir
end

(gx::GenerateContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = gx(c, flatten(args)...)
function (gx::GenerateContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        gx(generator.f, i)
    end
end

# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(trace), 
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

# ------------ Learnable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(trace),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, cl, w = generate(ss, ps, call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

@inline function (ctx::GenerateContext)(c::typeof(trace),
                                        addr::T,
                                        call::G,
                                        args...) where {G <: GenerativeFunction,
                                                        T <: Address}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    tr, w = generate(call, args, ss)
    ret = get_retval(tr)
    add_call!(ctx, addr, DynamicCallSite(get_choices(tr), get_score(tr), get_gen_fn(tr), get_args(tr), ret))
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function generate(opt::J, target::L, params, fn::Function, args...) where {J <: CompilationOptions, L <: AddressMap}
    ctx = Generate(opt, DynamicTrace(), target, params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(target::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Generate(DefaultPipeline(), DynamicTrace(), target, params)
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

function generate(target::L, fn::Function, args...) where L <: AddressMap
    ctx = Generate(DefaultPipeline(), DynamicTrace(), target, Empty())
    ret = ctx(fn, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, fn, args, ret), ctx.weight
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct GenerateContext{T <: AddressMap, K <: AddressMap, P <: Parameters} <: ExecutionContext
     tr::T
     target::K
     weight::Float64
     score::Float64
     visited::Visitor
     params::P
end
```
`GenerateContext` is used to generate traces, as well as record and accumulate likelihood weights given observations at addressed randomness.

Inner constructors:
```julia
GenerateContext(tr::T, target::K) where {T <: AddressMap, K <: AddressMap} = new{T, K}(tr, target, 0.0, Visitor(), Parameters())
GenerateContext(tr::T, target::K, params::P) where {T <: AddressMap, K <: AddressMap, P <: Parameters} = new{T, K, P}(tr, target, 0.0, Visitor(), params)
```
Outer constructors:
```julia
Generate(target::AddressMap) = GenerateContext(AddressMap(), target)
Generate(target::AddressMap, params) = GenerateContext(AddressMap(), target, params)
Generate(tr::AddressMap, target::AddressMap) = GenerateContext(tr, target)
```
""", GenerateContext)

@doc(
"""
```julia
ret, cl, w = generate(target::L, fn::Function, args...; params = Parameters()) where L <: AddressMap
ret, cs, w = generate(target::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: AddressMap, K}
ret, v_cl, w = generate(target::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(target::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: AddressMap
ret, v_cl, w = generate(target::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: AddressMap, K}
```

`generate` provides an API to the `GenerateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, a `RecordSite` instance specialized to the call, and the score/weight `w` computed with respect to the constraints `target`.
""", generate)

