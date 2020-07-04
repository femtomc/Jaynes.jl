mutable struct UpdateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    prev::T
    tr::T
    select::K
    visited::VisitedSelection
    UpdateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, Trace(), select, VisitedSelection())
end
Update(tr::Trace, select) = UpdateContext(tr, select)

# Update has a special dynamo.
@dynamo function (mx::UpdateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(rand), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.prev.chm, addr)
    in_prev_chm && begin
        prev = ctx.prev.chm[addr]
        prev_ret = prev.val
        prev_score = prev.score
    end

    # Check if in selection.
    in_selection = haskey(ctx.select.query, addr)

    # Ret.
    if in_selection
        ret = ctx.select.query[addr]
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        ctx.tr.score += score - prev_score
    elseif in_selection
        ctx.tr.score += score
    end
    ctx.tr.chm[addr] = ChoiceSite(score, ret)

    push!(ctx.visited, addr)
    return ret
end

# ------------ Call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(rand),
                                      addr::T,
                                      call::Function,
                                      args...) where T <: Address
    u_ctx = Update(ctx.prev.chm[addr].trace, ctx.select[addr])
    ret = u_ctx(call, args...)
    ctx.tr.chm[addr] = CallSite(u_ctx.tr, 
                                call, 
                                args, 
                                ret)
    ctx.visited.tree[addr] = u_ctx.visited
    return ret
end

