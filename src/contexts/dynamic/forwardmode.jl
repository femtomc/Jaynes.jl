# ------------ Choice sites ------------ #

@inline function (ctx::ForwardModeContext)(call::typeof(rand), 
                                           addr::A, 
                                           d::Distribution{K}) where {A <: Address, K}
    visit!(ctx, addr)
    val = Dual(getindex(ctx.target, addr), addr == ctx.addr[1] ? 1.0 : 0.0)
    ctx.weight += logpdf(d, val)
    return val
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

# TODO.
#@inline function (ctx::ForwardModeContext)(c::typeof(rand),
#                                           addr::A,
#                                           call::Function,
#                                           args...) where A <: Address
#    visit!(ctx, addr)
#    ps = get_sub(ctx.params, addr)
#    ss = get_sub(ctx.target, addr)
#    ret, w = score(ss, ps, call, args...) 
#    increment!(ctx, w)
#    return ret
#end

# ------------ Convenience ------------ #

function get_target_gradient(addr::T, cl::DynamicCallSite) where T <: Tuple
    arr = array(cl, Float64)
    fn = arr -> begin
        tg = target(get_trace(cl), arr)
        ctx = ForwardMode(addr, tg)
        ret = ctx(cl.fn, cl.args...)
        ctx.weight
    end
    grad = ForwardDiff.gradient(fn, arr)
    grad
end
