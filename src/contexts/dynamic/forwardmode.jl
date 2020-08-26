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
    fn = seed -> begin
        ctx = ForwardMode(addr, cl, seed)
        ret = ctx(cl.fn, cl.args...)
        ctx.weight
    end
    grad = gradient(1.0) do x
        Zygote.forwarddiff(x) do x
            fn(x)
        end
    end
    grad
end
