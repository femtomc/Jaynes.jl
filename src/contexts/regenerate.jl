mutable struct RegenerateContext{C <: CallSite,
                                 T <: Trace, 
                                 K <: UnconstrainedSelection,
                                 P <: Parameters,
                                 D <: Diff} <: ExecutionContext
    prev::C
    tr::T
    select::K
    weight::Float64
    score::Float64
    discard::HierarchicalTrace
    visited::Visitor
    params::P
    argdiffs::D
    RegenerateContext(cl::C, select::K, argdiffs::D) where {C <: CallSite, K <: UnconstrainedSelection, D <: Diff} = new{C, typeof(cl.trace), K, EmptyParameters, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, Trace(), Visitor(), Parameters(), argdiffs)
    RegenerateContext(cl::C, select::K, params::P, argdiffs::D) where {C <: CallSite, K <: UnconstrainedSelection, P <: Parameters, D <: Diff} = new{C, typeof(cl.trace), K, P, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, Trace(), Visitor(), params, argdiffs)
end
Regenerate(cl, select, argdiffs) = RegenerateContext(cl, select, argdiffs)

# Regenerate has a special dynamo.
@dynamo function (mx::RegenerateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

# ------------ Convenience ------------ #

function regenerate(ctx::RegenerateContext, bbcs::HierarchicalCallSite, new_args...)
    ret = ctx(bbcs.fn, new_args...)
    visited = ctx.visited
    adj_w = adjust_to_intersection(get_trace(bbcs), visited)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score - adj_w, bbcs.fn, new_args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, bbcs::HierarchicalCallSite) where L <: UnconstrainedSelection
    argdiffs = NoChange()
    ctx = RegenerateContext(bbcs, sel, argdiffs)
    return regenerate(ctx, bbcs, bbcs.args...)
end

function regenerate(sel::L, params, bbcs::HierarchicalCallSite) where L <: UnconstrainedSelection
    argdiffs = NoChange()
    ctx = RegenerateContext(bbcs, sel, params, argdiffs)
    return regenerate(ctx, bbcs, bbcs.args...)
end

function regenerate(sel::L, bbcs::HierarchicalCallSite, argdiffs::D, new_args...) where {L <: UnconstrainedSelection, D <: Diff}
    ctx = RegenerateContext(bbcs, sel, argdiffs)
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(sel::L, params, bbcs::HierarchicalCallSite, argdiffs::D, new_args...) where {L <: UnconstrainedSelection, D <: Diff}
    ctx = RegenerateContext(bbcs, sel, params, argdiffs)
    return regenerate(ctx, bbcs, new_args...)
end

# TODO: fix for dispatch with params.
function regenerate(sel::L, vcs::VectorizedCallSite{typeof(plate)}) where {L <: UnconstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = RegenerateContext(vcs, v_sel, argdiffs)
    ret = ctx(plate, addr, vcs.fn, vcs.args)
    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, vcs::VectorizedCallSite{typeof(markov)}) where {L <: UnconstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = RegenerateContext(vcs, v_sel, argdiffs)
    ret = ctx(markov, addr, vcs.fn, vcs.args[1], vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, vcs::VectorizedCallSite{typeof(markov)}, d::NoChange, len::Int) where {L <: UnconstrainedSelection, D <: Diff}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = RegenerateContext(vcs, v_sel, d)
    ret = ctx(markov, addr, vcs.fn, len, vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: UnconstrainedSelection, D <: Diff}
    return regenerate(sel, vcs, NoChange(), len)
end

# ------------ includes ------------ #

include("hierarchical/regenerate.jl")
include("plate/regenerate.jl")
include("markov/regenerate.jl")
include("factor/regenerate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct RegenerateContext{T <: Trace, 
                                 L <: UnconstrainedSelection,
                                 P <: Parameters} <: ExecutionContext
    prev::T
    tr::T
    select::L
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::P
end
```

Inner constructors:

```julia
function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
    un_sel = selection(sel)
    new{T, typeof(un_sel), EmptyParameters}(tr, Trace(), un_sel, 0.0, Trace(), Visitor(), Parameters())
end
function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
    new{T, L, EmptyParameters}(tr, Trace(), sel, 0.0, Trace(), Visitor(), Parameters())
end
```

Outer constructors:

```julia
Regenerate(tr::Trace, sel::Vector{Address}) = RegenerateContext(tr, sel)
Regenerate(tr::Trace, sel::UnconstrainedSelection) = RegenerateContext(tr, sel)
```

The `RegenerateContext` is used for MCMC algorithms, to propose new choices for addresses indicated by an `UnconstrainedSelection` in the `select` field.
""", RegenerateContext)

@doc(
"""
```julia
ret, cl = regenerate(sel::L, bbcs::HierarchicalCallSite, new_args...) where L <: UnconstrainedSelection
ret, cl = regenerate(sel::L, bbcs::HierarchicalCallSite) where L <: UnconstrainedSelection
```
`regenerate` is an API to the `RegenerateContext` execution context. `regenerate` requires that users provide an `UnconstrainedSelection`, an original call site, and possibly a set of new arguments to be used in the regeneration step. This context internally keeps track of the bookkeeping required to increment likelihood weights, as well as prune off parts of the trace which are invalid if a regenerated choice changes the shape of the trace (e.g. control flow), and returns a new return value `ret` as well as the modified call site `cl`.
""", regenerate)
