# TODO: specialize to different call sites.
mutable struct UpdateContext{C <: CallSite, 
                             T <: Trace, 
                             K <: ConstrainedSelection, 
                             P <: Parameters, 
                             D <: Diff} <: ExecutionContext
    prev::C
    tr::T
    select::K
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::P
    argdiff::D
    # Re-write with dispatch for specialized vs. black box.
    UpdateContext(cl::C, select::K, argdiffs::D) where {C <: CallSite, K <: ConstrainedSelection, D <: Diff} = new{C, typeof(cl.trace), K, NoParameters, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, Trace(), Visitor(), Parameters(), argdiffs)
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
    # Check if in previous trace's choice map.
    in_prev_chm = has_choice(ctx.prev.trace, addr)
    in_prev_chm && begin
        prev = get_choice(ctx.prev.trace, addr)
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

    has_addr = has_choice(ctx.prev.trace, addr)
    if has_addr
        cs = get_prev(ctx, addr)
        ss = get_subselection(ctx, addr)

        # TODO: Mjolnir.
        ret, new_site, lw, retdiff, discard = update(ss, cs, args...)

        add_choice!(ctx.discard, addr, CallSite(discard, cs.score, cs.fn, cs.args, cs.ret))
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

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, bbcs::GenericCallSite, args...)
    ret = ctx(bbcs.fn, args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, bbcs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, tr::T, fn::Function, new_args...) where {T <: Trace, L <: ConstrainedSelection}
    ctx = UpdateContext(tr, sel)
    ret = ctx(fn, new_args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, fn, new_args, ret), ctx.weight, UndefinedChange, ctx.discard
end

function update(sel::L, bbcs::GenericCallSite) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs, sel, NoChange())
    return update(ctx, bbcs, bbcs.args...)
end

function update(bbcs::GenericCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs, ConstrainedHierarchicalSelection())
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, bbcs::GenericCallSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(bbcs, sel, UndefinedChange())
    return update(ctx, bbcs, new_args...)
end

function update(argdiffs::D, bbcs::GenericCallSite, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs, ConstrainedHierarchicalSelection(), argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, argdiffs::D, bbcs::GenericCallSite, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, vcs::VectorizedSite, new_args...) where L <: ConstrainedSelection
    ctx = UpdateContext(vcs, sel)
    return update(ctx, vcs, new_args...)
end

function update(sel::L, argdiffs::D, vcs::VectorizedSite, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(vcs, sel, argdiffs)
    return update(ctx, vcs, new_args...)
end
