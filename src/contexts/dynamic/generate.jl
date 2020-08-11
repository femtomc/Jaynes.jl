# ------------ Choice sites ------------ #

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    if has_value(ctx.schema, addr)
        s = getindex(ctx.schema, addr)
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
    has_top(ctx.params, addr) && return get_top(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(fillable), addr::Address)
    has_top(ctx.schema, addr) && return get_top(ctx.schema, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::GenerateContext)(c::typeof(rand),
                                        addr::T,
                                        call::Function,
                                        args...) where T <: Address
    visit!(ctx, addr)
    ps = get_subparameters(ctx, addr)
    ss = get_subschema(ctx, addr)
    ret, cl, w = generate(ss, ps, call, args...)
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end
