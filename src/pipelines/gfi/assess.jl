# ------------ Staging ------------ #

@dynamo function (sx::AssessContext{J})(a...) where J
    ir = IR(a...)
    ir == nothing && return
    ir = pipeline(ir, AssessContext{J})
    ir
end

# Base fixes.
(sx::AssessContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = sx(c, flatten(args)...)
function (sx::AssessContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        sx(generator.f, i)
    end
end

# ------------ Choice sites ------------ #

@inline function (ctx::AssessContext)(call::typeof(trace), 
                                      addr::A, 
                                      d::Distribution{K}) where {A <: Address, K}
    visit!(ctx, addr)
    haskey(ctx.target, addr) || error("AssessError: constrained target must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = getindex(ctx.target, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Learnable ------------ #

@inline function (ctx::AssessContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::AssessContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::AssessContext)(c::typeof(trace),
                                      addr::A,
                                      call::Function,
                                      args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, w = assess(ss, ps, call, args...) 
    increment!(ctx, w)
    return ret
end

@inline function (ctx::AssessContext)(c::typeof(trace),
                                      addr::A,
                                      call::G,
                                      args...) where {G <: GenerativeFunction,
                                                      A <: Address}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    ret, w = assess(ss, ps, call.fn, args...) 
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function assess(opt::J, sel::L, params, fn::Function, args...) where {J <: CompilationOptions, L <: AddressMap}
    ctx = Assess(opt, sel, params)
    ret = ctx(fn, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("AssessError: did not visit all constraints in target.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function assess(sel::L, params, fn::Function, args...) where L <: AddressMap
    ctx = Assess(DefaultPipeline(), sel, params)
    ret = ctx(fn, args...)
    b, missed = compare(sel, ctx.visited)
    b || error("AssessError: did not visit all constraints in target.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct AssessContext{P <: AddressMap} <: ExecutionContext
    select::AddressMap
    weight::Float64
    params::P
end
```

The `AssessContext` is used to assess selections according to a model function. For computation in the `AssessContext` to execute successfully, the `select` selection must provide a choice for every address visited in the model function, and the model function must allow the context to visit every constraints expressed in `select`.

Inner constructors:

```julia
function Assess(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
    c_sel = selection(obs)
    new{EmptyAddressMap}(c_sel, 0.0, AddressMap())
end
```

Outer constructors:

```julia
AssessContext(obs::K, params) where {K <: AddressMap} = new(obs, 0.0, params)
Assess(obs::Vector) = AssessContext(selection(obs))
Assess(obs::AddressMap) = AssessContext(obs, AddressMap())
Assess(obs::AddressMap, params) = AssessContext(obs, params)
```
""", AssessContext)

@doc(
"""
```julia
ret, w = assess(sel::L, fn::Function, args...; params = AddressMap()) where L <: AddressMap
ret, w = assess(sel::L, fn::typeof(rand), d::Distribution{K}; params = AddressMap()) where {L <: AddressMap, K}
```

`assess` provides an API to the `AssessContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, and the likelihood weight assess of the user-provided selection `sel`. The selection should satisfy the following requirement:

1. At any random choice in any branch traveled according to the constraints of `sel`, `sel` must provide a constraint for that choice.

Simply put, this just means you need to provide a constraint for each `ChoiceSite` you encounter.
""", assess)
