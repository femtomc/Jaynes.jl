mutable struct RegenerateContext{T <: Trace, L <: UnconstrainedSelection} <: ExecutionContext
    prev::T
    tr::T
    select::L
    discard::ConstrainedHierarchicalSelection
    visited::VisitedSelection
    function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
        un_sel = selection(sel)
        new{T, typeof(un_sel)}(tr, Trace(), un_sel, ConstrainedHierarchicalSelection(), VisitedSelection())
    end
    function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
        new{T, L}(tr, Trace(), sel, ConstrainedHierarchicalSelection(), VisitedSelection())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}) = RegenerateContext(tr, sel)
Regenerate(tr::Trace, sel::UnconstrainedSelection) = RegenerateContext(tr, sel)

# Regenerate has a special dynamo.
@dynamo function (mx::RegenerateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(rand), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.prev.chm, addr)

    # Check if in selection in meta.
    in_sel = haskey(ctx.select.query, addr)

    if in_prev_chm
        prev = ctx.prev.chm[addr]
        if in_sel
            ret = rand(d)
            push!(ctx.discard, addr, prev.val)
        else
            ret = prev.val
        end
    end

    score = logpdf(d, ret)
    if in_prev_chm && !in_sel
        ctx.tr.score += score - prev.score
    end

    ctx.tr.chm[addr] = ChoiceSite(score, ret)
    push!(ctx.visited, addr)
    return ret
end

# ------------ Call sites ------------ #

@inline function (ctx::RegenerateContext)(c::typeof(rand),
                                          addr::T,
                                          call::Function,
                                          args...) where T <: Address
    ur_ctx = Regenerate(ctx.prev.chm[addr].trace, ctx.select[addr])
    ret = ur_ctx(call, args...)
    ctx.tr.chm[addr] = BlackBoxCallSite(ur_ctx.tr, 
                                        call, 
                                        args, 
                                        ret)
    ctx.visited.tree[addr] = ur_ctx.visited
    return ret
end

# Convenience.
function regenerate(ctx::RegenerateContext, bbcs::BlackBoxCallSite, new_args...)
    ctx(bbcs.fn, new_args...)
    return ctx.tr, ctx.tr.score, UndefinedChange(), ctx.discard
end

function regenerate(sel::L, bbcs::BlackBoxCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, sel)
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(bbcs::BlackBoxCallSite, new_args...) where L <: UnconstrainedSelection
    ctx = RegenerateContext(bbcs.trace, ConstrainedHierarchicalSelection())
    return regenerate(ctx, bbcs, new_args...)
end

function regenerate(ctx::RegenerateContext, vcs::VectorizedCallSite, new_args...)
end

function regenerate(sel::L, vcs::VectorizedCallSite, new_args...) where L <: UnconstrainedSelection
end
