mutable struct UpdateContext{T <: Trace, K <: ConstrainedSelection, D <: Diff} <: ExecutionContext
    prev::T
    tr::T
    select::K
    weight::Float64
    discard::T
    visited::Visitor
    params::LearnableParameters
    argdiff::D
    UpdateContext(tr::T, select::K, argdiffs::D) where {T <: Trace, K <: ConstrainedSelection, D <: Diff} = new{T, K, D}(tr, Trace(), select, 0.0, Trace(), Visitor(), LearnableParameters(), argdiffs)
end
Update(tr::Trace, select, argdiffs) = UpdateContext(tr, select, argdiffs)
Update(tr::Trace, select) = UpdateContext(tr, select, UndefinedChange())
Update(select) = UpdateContext(Trace(), select, UndefinedChange())

# Update has a special dynamo.
@dynamo function (mx::UpdateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(rand), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    # Check if in previous trace's choice plate.
    in_prev_chm = has_choice(ctx.prev, addr)
    in_prev_chm && begin
        prev = get_choice(ctx.prev, addr)
        prev_ret = prev.val
        prev_score = prev.score
    end

    # Check if in selection.
    in_selection = has_query(ctx.select, addr)

    # Ret.
    if in_selection
        ret = get_query(ctx.select, addr)
        in_prev_chm && begin
            add_choice!(ctx.discard, addr, prev)
        end
        visit!(ctx.visited, addr)
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        increment!(ctx, score - prev_score)
    elseif in_selection
        increment!(ctx, score)
    end
    add_choice!(ctx.tr, addr, ChoiceSite(score, ret))

    return ret
end

# ------------ Black box call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(rand),
                                      addr::T,
                                      call::Function,
                                      args...) where {T <: Address, D <: Diff}

    has_addr = has_choice(ctx.prev, addr)
    if has_addr
        cs = get_prev(ctx, addr)
        ss = get_subselection(ctx, addr)

        # TODO: Mjolnir.
        ret, new_site, lw, retdiff, discard = update(ss, cs, args...)

        add_choice!(ctx.discard, addr, CallSite(discard, cs.fn, cs.args, cs.ret))
    else
        ss = get_subselection(ctx, addr)
        ret, new_site, lw = generate(ss, call, args...)
    end
    add_call!(ctx.tr, addr, new_site)
    increment!(ctx, w)
    return ret
end

# ------------ Vectorized call sites ------------ #

# Vectorized convenience functions for plate.

@inline function (ctx::UpdateContext)(c::typeof(plate), 
                                      fn::typeof(rand), 
                                      addr::Address, 
                                      call::Function, 
                                      args::Vector)
    local sc_adj::Float64
    local new::Vector{CallSite}

    vcs = get_prev(ctx, addr)
    n_len, o_len = length(args), length(vcs.args)

    # Get targeted for update.
    ks = keyset(ctx.select, n_len)

    # Get score adjustment if vector is reduced in length, otherwise generate new traces which have selection applied.
    if n_len < o_len
        sc_adj = -sum(map(vcs.subcalls[n_len : end]) do cl
                         get_score(cl)
                     end)
        new = vcs.subcalls[1 : n_len]
        new_ret = typeof(vcs.ret)(undef, n_len)
        for i in 1 : n_len
            i in ks && begin
                s = get_subselection(ctx, addr => i)
                ret, u_cl, u_w, rd, d = update(s, vcs.subcalls[i], args[i]...)
                new_ret[i] = ret
                new[i] = u_cl
                sc_adj += u_w - get_score(vcs.subcalls[i])
            end
        end
   
    # Otherwise, generate new with constraints from o_len to n_len, then call update on the old call sites.
    else
        new = Vector{CallSite}(undef, n_len)
        new_ret = typeof(vcs.ret)(undef, n_len)
        for i in o_len : n_len
            s = get_subselection(ctx, addr => i)
            ret, cl, w = generate(s, fn, args[i]...)
            new_ret[i] = ret
            new[i] = cl
            sc_adj += w
        end
        sc_adj = 0.0
        for i in 1 : o_len
            i in ks && begin
                s = get_subselection(ctx, addr => i)
                ret, u_cl, u_w, rd, d = update(s, vcs.subcalls[i], args[i]...)
                new_ret[i] = ret
                new[i] = u_cl
                score_adj += u_w - get_score(vcs.subcalls[i])
                continue
            end
            new[i] = vcs.subcalls[i]
        end
    end

    # Add new vectorized site.
    add_call!(ctx.tr, addr, VectorizedSite{typeof(plate)}(new, vcs.score + sc_adj, vcs.fn, args, new_ret))
    return new_ret
end

# Vectorized convenience functions for markov.

function retrace_retained(addr::Address,
                          new_trs::Vector{T},
                          vcs::VectorizedSite{typeof(markov), T, J, K}, 
                          sel::L, 
                          args::Vector,
                          targeted::Set{Int},
                          prev_length::Int,
                          new_length::Int) where {T <: Trace, J, K, L <: ConstrainedSelection}
    u_ctx = Update(sel[addr][1])
    updated = Vector{T}(undef, new_length)
    ret = typeof(vcs.ret)(undef, new_length)
    for k in 1 : min(prev_length, new_length)
        u_ctx.prev = new_trs[k]
        u_ctx.select = sel[addr][k]
        ret = u_ctx(vcs.fn, args[k]...)
        ret[k] = ret
        updated[k] = u_ctx.tr
    end
    score_adj = sum(map(updated) do tr
                        score(tr)
                    end)
    return score_adj, updated, ret
end

@inline function (ctx::UpdateContext)(c::typeof(markov), 
                                      fn::typeof(rand), 
                                      addr::Address, 
                                      call::Function, 
                                      len::Int,
                                      args...)
    # Grab VCS and compute new and old lengths.
    vcs = ctx.prev.chm[addr]
    n_len, o_len = len, length(vcs.subtraces)

    # Get indices to subset of traces which are in selection.
    ks = keyset(ctx.select, len)

    # Adjust score by computing cumulative score for traces which are at indices less than the new length.
    sc_adj = score_adj(ctx.prev.chm[addr], o_len, n_len)
    vcs.score -= sc_adj

    # Get the set of traces which are retained.
    retained = vcs.subtraces[n_len + 1 : o_len]

    # Generate new traces (if n_len > o_len).

    return new_ret
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, bbcs::BlackBoxCallSite, args...)
    ret = ctx(bbcs.fn, args...)
    return ret, BlackBoxCallSite(ctx.tr, bbcs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, tr::T, fn::Function, new_args...) where {T <: Trace, L <: ConstrainedSelection}
    ctx = UpdateContext(tr, sel)
    ret = ctx(fn, new_args...)
    return ret, BlackBoxCallSite(ctx.tr, fn, new_args, ret), ctx.weight, UndefinedChange, ctx.discard
end

function update(sel::L, bbcs::BlackBoxCallSite) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, sel, NoChange())
    return update(ctx, bbcs, bbcs.args...)
end

function update(bbcs::BlackBoxCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, ConstrainedHierarchicalSelection())
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, bbcs::BlackBoxCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs.trace, sel, UndefinedChange())
    return update(ctx, bbcs, new_args...)
end

function update(argdiffs::D, bbcs::BlackBoxCallSite, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs.trace, ConstrainedHierarchicalSelection(), argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, argdiffs::D, bbcs::BlackBoxCallSite, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs.trace, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(ctx::UpdateContext, vcs::VectorizedSite, new_args...)
end

function update(ctx::UpdateContext, vcs::VectorizedSite, new_args...)
end

function update(sel::L, vcs::VectorizedSite, new_args...) where L <: ConstrainedSelection
end

function update(sel::L, vcs::VectorizedSite, new_args...) where L <: ConstrainedSelection
end
