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

@inline function (ctx::SimulateContext)(fn::typeof(learnable), addr::T) where T <: Address
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::SimulateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ret, cl = simulate(ps, call, args...)
    add_call!(ctx, addr, cl)
    return ret
end
