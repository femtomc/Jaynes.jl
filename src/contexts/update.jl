mutable struct UpdateContext{T <: Trace, K <: ConstrainedSelection} <: ExecutionContext
    prev::T
    tr::T
    select::K
    discard::ConstrainedHierarchicalSelection
    visited::VisitedSelection
    UpdateContext(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, Trace(), select, ConstrainedHierarchicalSelection(), VisitedSelection())
end
Update(tr::Trace, select) = UpdateContext(tr, select)
Update(select) = UpdateContext(Trace(), select)

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
        in_prev_chm && push!(ctx.discard, addr, prev_ret)
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
   
    has_addr = haskey(ctx.prev, addr)
    if has_addr
        cs = ctx.prev.chm[addr]

        # TODO: Mjolnir.
        new_site, lw, _, discard = update(cs, ctx.select[addr], args...; diffs = map((_) -> UndefinedChange(), args))

        push!(ctx.discard, addr, discard)
    else
        new_site, lw = generate(call, ctx.select[addr], args...)
    end
    ctx.tr.chm[addr] = new_site
    ctx.tr.score += lw
    return new_site.ret
end

# Vectorized convenience functions for map.
function map_retrace_retained(addr::Address,
                              new_trs::Vector{T},
                              vcs::VectorizedCallSite{T, J, K}, 
                              sel::L, 
                              args::Vector,
                              targeted::Set{Int},
                              prev_length::Int,
                              new_length::Int) where {T <: Trace, J, K, L <: ConstrainedSelection}
    u_ctx = Update()
    updated = Vector{T}(undef, new_length)
    ret = typeof(vcs.ret)(undef, new_length)
    for k in 1 : min(prev_length, new_length)
        u_ctx.prev = new_trs[k]
        u_ctx.sel = sel[addr => k]
        ret = u_ctx(vcs.fn, args[k]...)
        ret[k] = ret
        updated[k] = u_ctx.tr
    end
    score_adj = sum(map(updated) do tr
                        score(tr)
                    end)
    return score_adj, updated, ret
end

function map_generate_new(addr::Address, 
                          vcs::VectorizedCallSite{T, J, K}, 
                          sel::L,
                          args::Vector, 
                          prev_length::Int,
                          new_length::Int) where {T <: Trace, J, K, L <: ConstrainedSelection}
    if prev_length + 1 < new_length
        g_ctx = Generate()
        new_trs = Vector{T}(undef, new_length - prev_length)
        for k in prev_length + 1 : new_length
            g_ctx.sel = sel[addr => k]
            g_ctx(vcs.fn, args...)
            new_trs[k - prev_length - 1] = g_ctx.tr
            g_ctx.tr = Trace()
        end
        return new_trs
    end
    return Vector{T}()
end

@inline function (ctx::UpdateContext)(c::typeof(map), 
                                      fn::typeof(rand), 
                                      addr::Address, 
                                      call::Function, 
                                      args::Vector)
    # Grab VCS and compute new and old lengths.
    vcs = ctx.prev[addr]
    n_len, o_len = length(args), length(vcs.args)

    # Get indices to subset of traces which are in selection.
    ks = keyset(ctx.sel, n_len)

    # Adjust score by computing cumulative score for traces which are at indices less than the new length.
    sc_adj = score_adjust(ctx.tr[addr], o_len, n_len)
    vcs.score -= sc_adj

    # Get the set of traces which are retained.
    retained = vcs.subtraces[n_len + 1 : o_len]

    # Generate new traces (if n_len > o_len).
    new_trs = map_generate_new(addr, vcs, sel, args, o_len, n_len)

    # Append to the set of retained traces.
    append!(retained, new_trs)

    # Update the total set of traces, calculate the score adjustment and new returns.
    sc_adj, updated, new_ret = map_retrace_retained(addr, retained, vcs, sel, args, ks, o_len, n_len)

    # Create a new VectorizedCallSite.
    ctx.tr[addr] = VectorizedCallSite(retained, vcs.score - sc_adj, vcs.fn, args, new_ret)

    return new_ret
end

# Vectorized convenience functions for foldr.
function map_retrace_retained(addr::Address,
                              new_trs::Vector{T},
                              vcs::VectorizedCallSite{T, J, K}, 
                              sel::L, 
                              args::Vector,
                              targeted::Set{Int},
                              prev_length::Int,
                              new_length::Int) where {T <: Trace, J, K, L <: ConstrainedSelection}
    u_ctx = Update()
    updated = Vector{T}(undef, new_length)
    ret = typeof(vcs.ret)(undef, new_length)
    for k in 1 : min(prev_length, new_length)
        u_ctx.prev = new_trs[k]
        u_ctx.select = sel[addr => k]
        ret = u_ctx(vcs.fn, args[k]...)
        ret[k] = ret
        updated[k] = u_ctx.tr
    end
    score_adj = sum(map(updated) do tr
                        score(tr)
                    end)
    return score_adj, updated, ret
end

@inline function (ctx::UpdateContext)(c::typeof(foldr), 
                                      fn::typeof(rand), 
                                      addr::Address, 
                                      call::Function, 
                                      len::Int,
                                      args...)
    # Grab VCS and compute new and old lengths.
    vcs = ctx.prev[addr]
    n_len, o_len = len, length(vcs.subtraces)

    # Get indices to subset of traces which are in selection.
    ks = keyset(ctx.select, len)

    # Adjust score by computing cumulative score for traces which are at indices less than the new length.
    sc_adj = score_adjust(ctx.tr[addr], o_len, n_len)
    vcs.score -= sc_adj

    # Get the set of traces which are retained.
    retained = vcs.subtraces[n_len + 1 : o_len]

    # Generate new traces (if n_len > o_len).

    return new_ret
end

# Convenience.
function update(ctx::UpdateContext, bbcs::BlackBoxCallSite, args...)
    ret = ctx(bbcs.fn, args...)
    return BlackBoxCallSite(ctx.tr, bbcs.fn, args, ret), UndefinedChange(), ctx.discard
end

function update(sel::L, bbcs::BlackBoxCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, sel)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, bbcs::BlackBoxCallSite) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, sel)
    return update(ctx, bbcs, bbcs.args...)
end

function update(bbcs::BlackBoxCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, ConstrainedHierarchicalSelection())
    return update(ctx, bbcs, new_args...)
end

function update(ctx::UpdateContext, vcs::VectorizedCallSite, new_args...)
end

function update(sel::L, vcs::VectorizedCallSite, new_args...) where L <: ConstrainedSelection
end
