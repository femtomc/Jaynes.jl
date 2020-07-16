mutable struct ScoreContext{P <: Parameters} <: ExecutionContext
    select::ConstrainedSelection
    weight::Float64
    visited::Visitor
    params::P
    function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
        c_sel = selection(obs)
        new{NoParameters}(c_sel, 0.0, Parameters())
    end
    ScoreContext(obs::K, params::P) where {K <: ConstrainedSelection, P <: Parameters} = new{P}(obs, 0.0, Visitor(), params)
end
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::ConstrainedSelection) = ScoreContext(obs, Parameters())
Score(obs::ConstrainedSelection, params) = ScoreContext(obs, params)

# ------------ Convenience ------------ #

function score(sel::L, fn::Function, args...; params = Parameters()) where L <: ConstrainedSelection
    ctx = Score(sel, params)
    ret = ctx(fn, args...)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(rand), d::Distribution{K}; params = Parameters()) where {L <: ConstrainedSelection, K}
    ctx = Score(sel, params)
    addr = gensym()
    ret = ctx(fn, addr, d)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(markov), call::Function, len::Int, args...; params = Parameters()) where L <: ConstrainedSelection
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Score(v_sel, params)
    ret = ctx(fn, addr, call, len, args...)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(plate), call::Function, args::Vector; params = Parameters()) where L <: ConstrainedSelection
    ctx = Score(sel, params)
    addr = gensym()
    ret = ctx(fn, addr, call, args)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

function score(sel::L, fn::typeof(plate), d::Distribution{K}, len::Int; params = Parameters()) where {L <: ConstrainedSelection, K}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = Score(v_sel, params)
    ret = ctx(fn, addr, d, len)
    b, missed = compare(sel.query, ctx.visited)
    b || error("ScoreError: did not visit all constraints in selection.\nDid not visit: $(missed).")
    return ret, ctx.weight
end

# ------------ includes ------------ #

include("generic/score.jl")
include("plate/score.jl")
include("markov/score.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct ScoreContext{P <: Parameters} <: ExecutionContext
    select::ConstrainedSelection
    weight::Float64
    params::P
end
```

The `ScoreContext` is used to score selections according to a model function. For computation in the `ScoreContext` to execute successfully, the `select` selection must provide a choice for every address visited in the model function, and the model function must allow the context to visit every constraints expressed in `select`.

Inner constructors:

```julia
function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
    c_sel = selection(obs)
    new{NoParameters}(c_sel, 0.0, Parameters())
end
```

Outer constructors:

```julia
ScoreContext(obs::K, params) where {K <: ConstrainedSelection} = new(obs, 0.0, params)
Score(obs::Vector) = ScoreContext(selection(obs))
Score(obs::ConstrainedSelection) = ScoreContext(obs, Parameters())
Score(obs::ConstrainedSelection, params) = ScoreContext(obs, params)
```
""", ScoreContext)
