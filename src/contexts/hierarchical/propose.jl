# ------------ Choice sites ------------ #

@inline function (ctx::ProposeContext)(call::typeof(rand), 
                                       addr::T, 
                                       d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    s = rand(d)
    score = logpdf(d, s)
    add_choice!(ctx, addr, ChoiceSite(score, s))
    return s
end

# ------------ Learnable ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::ProposeContext)(c::typeof(rand),
                                       addr::T,
                                       call::Function,
                                       args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ret, cl, w = propose(ps, call, args...)
    add_call!(ctx, addr, cl)
    return ret
end
