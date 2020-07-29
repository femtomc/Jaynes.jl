# ------------ Choice sites ------------ #

@inline function (ctx::SimulateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx.visited, addr)
    s = rand(d)
    add_choice!(ctx, addr, ChoiceSite(logpdf(d, s), s))
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::SimulateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_param(ctx.params, addr) && return get_param(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    ret, cl = simulate(call, args...)
    add_call!(ctx, addr, cl)
    return ret
end

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args::Tuple,
                                        ret_score::Function) where T <: Address
    visit!(ctx, addr)
    ret, cl, w = simulate(call, args...)
    add_call!(ctx, addr, cl, ret_score(ret))
    return ret
end

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args::Tuple,
                                        ret_score::Distribution{K}) where {K, T <: Address}
    visit!(ctx, addr)
    ret, cl = simulate(call, args...)
    add_call!(ctx, addr, cl, logpdf(ret_score, ret))
    return ret
end
