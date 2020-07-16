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
    function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
        un_sel = selection(sel)
        new{T, typeof(un_sel), NoParameters}(tr, Trace(), un_sel, 0.0, 0.0, Trace(), Visitor(), Parameters())
    end
    function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
        new{T, L, NoParameters}(tr, Trace(), sel, 0.0, 0.0, Trace(), Visitor(), Parameters())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}) = RegenerateContext(tr, sel)
Regenerate(tr::Trace, sel::UnconstrainedSelection) = RegenerateContext(tr, sel)
get_prev(ctx::RegenerateContext, addr) = get_call(ctx.prev, addr)

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
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, bbcs.fn, new_args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, bbcs::HierarchicalCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, sel)
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(sel::L, bbcs::HierarchicalCallSite) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, sel)
    return regenerate(ctx, bbcs, bbcs.args...)
end

function regenerate(bbcs::HierarchicalCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, ConstrainedHierarchicalSelection())
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(ctx::RegenerateContext, vcs::VectorizedSite, new_args...)
end

function regenerate(sel::L, vcs::VectorizedSite, new_args...) where L <: UnconstrainedSelection
end

# ------------ includes ------------ #

include("generic/regenerate.jl")
include("plate/regenerate.jl")
include("markov/regenerate.jl")

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
    new{T, typeof(un_sel), NoParameters}(tr, Trace(), un_sel, 0.0, Trace(), Visitor(), Parameters())
end
function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
    new{T, L, NoParameters}(tr, Trace(), sel, 0.0, Trace(), Visitor(), Parameters())
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
ret, call_site = regenerate(sel::L, bbcs::HierarchicalCallSite, new_args...) where L <: UnconstrainedSelection
ret, call_site = regenerate(sel::L, bbcs::HierarchicalCallSite) where L <: UnconstrainedSelection
```
Users will likely interact with the `RegenerateContext` through the convenience method `regenerate` which requires that users provide an `UnconstrainedSelection`, an original call site, and possibly a set of new arguments to be used in the regeneration step. This context internally keeps track of the bookkeeping required to increment likelihood weights, as well as prune off parts of the trace which are invalid if a regenerated choice changes the shape of the trace (e.g. control flow).
""", regenerate)
