mutable struct ScoreContext <: ExecutionContext
    select::ConstrainedSelection
    weight::Float64
    params::LearnableParameters
    function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
        c_sel = selection(obs)
        new(c_sel, 0.0, LearnableParameters())
    end
    ScoreContext(obs::K) where {K <: ConstrainedSelection} = new(obs, 0.0, LearnableParameters())
end
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::ConstrainedSelection) = ScoreContext(obs)

# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    has_query(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = get_query(ctx.select, addr)
    ctx.weight += logpdf(d, val)
    return val
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    s_ctx = Score(ctx.select[addr])
    ret = s_ctx(call, args...)
    ctx.weight += s_ctx.weight
    return ret
end

@inline function (ctx::ScoreContext)(c::typeof(foldr), 
                                     fn::typeof(rand), 
                                     addr::Address, 
                                     call::Function, 
                                     len::Int, 
                                     args...)
    s_ctx = Score(ctx.select[addr => 1])
    ret = s_ctx(call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    for i in 2:len
        s_ctx.select = ctx.select[addr => i]
        s_ctx.tr = Trace()
        ret = s_ctx(call, v_ret[i-1]...)
        v_ret[i] = ret
    end
    return v_ret
end

@inline function (ctx::ScoreContext)(c::typeof(map), 
                                     fn::typeof(rand), 
                                     addr::Address, 
                                     call::Function, 
                                     args::Vector)
    s_ctx = Score(ctx.select[addr => 1])
    ret = s_ctx(call, args[1]...)
    len = length(args)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    for i in 2:len
        s_ctx.select = ctx.select[addr => i]
        ret = s_ctx(call, args[i]...)
        v_ret[i] = ret
    end
    return v_ret
end

# Convenience.
function score(sel::L, fn::Function, args...) where L <: ConstrainedSelection
    ctx = Score(sel)
    ret = ctx(fn, args...)
    return ctx.weight
end
