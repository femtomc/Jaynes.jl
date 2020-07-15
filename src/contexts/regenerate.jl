mutable struct RegenerateContext{T <: Trace, L <: UnconstrainedSelection} <: ExecutionContext
    prev::T
    tr::T
    select::L
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::LearnableParameters
    function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
        un_sel = selection(sel)
        new{T, typeof(un_sel)}(tr, Trace(), un_sel, 0.0, 0.0, Trace(), Visitor(), LearnableParameters())
    end
    function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
        new{T, L}(tr, Trace(), sel, 0.0, 0.0, Trace(), Visitor(), LearnableParameters())
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

# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(rand), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    visit!(ctx.visited, addr)
    in_prev_chm = has_choice(ctx.prev, addr)
    in_sel = has_query(ctx.select, addr)
    if in_prev_chm
        prev = get_choice(ctx.prev, addr)
        if in_sel
            ret = rand(d)
            add_choice!(ctx.discard, addr, prev)
        else
            ret = prev.val
        end
    end
    score = logpdf(d, ret)
    if in_prev_chm && !in_sel
        increment!(ctx, score - prev.score)
    end
    add_choice!(ctx, addr, ChoiceSite(score, ret))
    return ret
end
# ------------ Learnable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(learnable), addr::Address, p::T) where T
    visit!(ctx, addr)
    ret = p
    if has_param(ctx.params, addr)
        ret = get_param(ctx.params, addr)
    end
    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::RegenerateContext)(c::typeof(rand),
                                          addr::T,
                                          call::Function,
                                          args...) where T <: Address
    visit!(ctx, addr)
    ss = get_subselection(ctx, addr)
    prev_call = get_prev(ctx, addr)
    ret, cl, w, retdiff, d = regenerate(ss, prev_call, args...)
    set_call!(ctx.tr, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function regenerate(ctx::RegenerateContext, bbcs::GenericCallSite, new_args...)
    ret = ctx(bbcs.fn, new_args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, bbcs.fn, new_args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, bbcs::GenericCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, sel)
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(sel::L, bbcs::GenericCallSite) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, sel)
    return regenerate(ctx, bbcs, bbcs.args...)
end

function regenerate(bbcs::GenericCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, ConstrainedHierarchicalSelection())
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(ctx::RegenerateContext, vcs::VectorizedSite, new_args...)
end

function regenerate(sel::L, vcs::VectorizedSite, new_args...) where L <: UnconstrainedSelection
end

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct RegenerateContext{T <: Trace, L <: UnconstrainedSelection} <: ExecutionContext
    prev::T
    tr::T
    select::L
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::LearnableParameters
end
```

Inner constructors:

```julia
function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
    un_sel = selection(sel)
    new{T, typeof(un_sel)}(tr, Trace(), un_sel, 0.0, Trace(), Visitor(), LearnableParameters())
end
function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
    new{T, L}(tr, Trace(), sel, 0.0, Trace(), Visitor(), LearnableParameters())
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
ret, call_site = regenerate(sel::L, bbcs::GenericCallSite, new_args...) where L <: UnconstrainedSelection
ret, call_site = regenerate(sel::L, bbcs::GenericCallSite) where L <: UnconstrainedSelection
```
Users will likely interact with the `RegenerateContext` through the convenience method `regenerate` which requires that users provide an `UnconstrainedSelection`, an original call site, and possibly a set of new arguments to be used in the regeneration step. This context internally keeps track of the bookkeeping required to increment likelihood weights, as well as prune off parts of the trace which are invalid if a regenerated choice changes the shape of the trace (e.g. control flow).
""", regenerate)
