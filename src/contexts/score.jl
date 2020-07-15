mutable struct ScoreContext <: ExecutionContext
    select::ConstrainedSelection
    weight::Float64
    params::LearnableParameters
    function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
        c_sel = selection(obs)
        new(c_sel, 0.0, LearnableParameters())
    end
    ScoreContext(obs::K, params) where {K <: ConstrainedSelection} = new(obs, 0.0, params)
end
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::ConstrainedSelection) = ScoreContext(obs, LearnableParameters())
Score(obs::ConstrainedSelection, params) = ScoreContext(obs, params)

# ------------ Choice sites ------------ #

@inline function (ctx::ScoreContext)(call::typeof(rand), 
                                     addr::T, 
                                     d::Distribution{K}) where {T <: Address, K}
    has_query(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
    val = get_query(ctx.select, addr)
    increment!(ctx, logpdf(d, val))
    return val
end

# ------------ Call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(rand),
                                     addr::T,
                                     call::Function,
                                     args...) where T <: Address
    ss = get_subselection(ctx, addr)
    ret, w = score(ss, call, args...) 
    increment!(ctx, w)
    return ret
end

# ------------ Vectorized call sites ------------ #

@inline function (ctx::ScoreContext)(c::typeof(markov), 
                                     addr::Address, 
                                     call::Function, 
                                     len::Int, 
                                     args...)
    ss = get_subselection(ctx, addr => 1)
    ret, w = score(ss, call, args...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        ss = get_subselection(ctx, addr => i)
        ret, w = score(ss, call, v_ret[i-1]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

@inline function (ctx::ScoreContext)(c::typeof(plate), 
                                     addr::Address, 
                                     call::Function, 
                                     args::Vector)
    ss = get_subselection(ctx, addr => 1)
    len = length(args)
    ret, w = score(ss, call, args[1]...)
    v_ret = Vector{typeof(ret)}(undef, len)
    v_ret[1] = ret
    increment!(ctx, w)
    for i in 2:len
        ss = get_subselection(ctx, addr => i)
        ret, w = score(ss, call, args[i]...)
        v_ret[i] = ret
        increment!(ctx, w)
    end
    return v_ret
end

# ------------ Convenience ------------ #

function score(sel::L, fn::Function, args...; params = LearnableParameters()) where L <: ConstrainedSelection
    ctx = Score(sel, params)
    ret = ctx(fn, args...)
    return ret, ctx.weight
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ScoreContext <: ExecutionContext
    select::ConstrainedSelection
    weight::Float64
    params::LearnableParameters
end
```

The `ScoreContext` is used to score selections according to a model function. For computation in the `ScoreContext` to execute successfully, the `select` selection must provide a choice for every address visited in the model function.

Inner constructors:

```julia
function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
    c_sel = selection(obs)
    new(c_sel, 0.0, LearnableParameters())
end
```

Outer constructors:

```julia
ScoreContext(obs::K, params) where {K <: ConstrainedSelection} = new(obs, 0.0, params)
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::ConstrainedSelection) = ScoreContext(obs, LearnableParameters())
Score(obs::ConstrainedSelection, params) = ScoreContext(obs, params)
```
""", ScoreContext)
