mutable struct ScoreContext <: ExecutionContext
    score::Float64
    select::ConstrainedSelection
    function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
        c_sel = selection(obs)
        new(0.0, c_sel)
    end
    ScoreContext(obs::K) where {K <: ConstrainedSelection} = new(0.0, obs)
    Score() = new(0.0)
end
Score(obs::Vector) = ScoreContext(obs)
Score(obs::ConstrainedSelection) = ScoreContext(sel)
Score() = ScoreContext()

# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    haskey(ctx.select.query, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.") && begin
        val = ctx.select.query[addr]
    end
    ctx.score += logpdf(d, val)
    return val
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    s_ctx = Score(ctx.select[addr])
    ret = s_ctx(call, args...)
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
