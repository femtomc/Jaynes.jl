# ------------ Choice sites ------------ #

@inline function (ctx::ForwardModeContext)(call::typeof(rand), 
                                           addr::A, 
                                           d::Distribution{K}) where {A <: Address, K}
    visit!(ctx, addr)
    v = context_getindex(ctx, ctx.map, addr)
    ctx.weight += logpdf(d, v)
    v
end

# ------------ Learnable ------------ #

@inline function (ctx::ForwardModeContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::ForwardModeContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ForwardModeContext)(c::typeof(rand),
                                           addr::A,
                                           call::Function,
                                           args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, w = forward(ctx.target[2 : end], ps, get_sub(ctx.map, addr), Dual(1.0, 0.0))
    ctx.weight += w
    return ret
end

# ------------ Convenience ------------ #

function get_target_gradient(addr::T, cl::DynamicCallSite) where T <: Tuple
    fn = seed -> begin
        ret, w = forward(addr, Empty(), cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    cl[addr], d.partials.values[1]
end
