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
    UpdateContext(cl::C, select::K, argdiffs::D) where {C <: CallSite, K <: ConstrainedSelection, D <: Diff} = new{C, typeof(cl.trace), K, NoParameters, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, typeof(cl.trace)(), Visitor(), Parameters(), argdiffs)
end

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
    increment!(ctx, lw)
    return ret
end

# ------------ Vectorized call sites ------------ #

# Vectorized convenience functions for plate.

@inline function (ctx::UpdateContext)(c::typeof(plate), 
                                      addr::Address, 
                                      call::Function, 
                                      args::Vector)
    local sc_adj::Float64

    if has_call(ctx.prev.trace, addr) 
        vcs = get_prev(ctx, addr)
    else
        vcs = ctx.prev
    end
    n_len, o_len = length(args), length(vcs.args)
    ss = get_subselection(ctx, addr)

    # Get targeted for update.
    ks = keyset(ss, n_len)

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
        sc_adj = 0.0
        new_ret = typeof(vcs.ret)(undef, n_len)
        new = vcs.trace.subrecords
        for i in o_len : n_len
            s = get_subselection(ctx, addr => i)
            ret, cl, w = generate(s, call, args[i]...)
            new_ret[i] = ret
            new[i] = cl
            sc_adj += w
        end
        for i in 1 : o_len
            i in ks && begin
                s = get_subselection(ctx, addr => i)
                ret, u_cl, u_w, rd, d = update(s, get_call(vcs, i), NoChange(), args[i]...)
                new_ret[i] = ret
                new[i] = u_cl
                sc_adj += u_w - get_score(get_call(vcs, i))
                continue
            end
        end
    end

    # Add new vectorized site.
    for n in new
        add_call!(ctx.tr, n)
    end
    increment!(ctx, sc_adj)
    return new_ret
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, bbcs::GenericCallSite, args...) where D <: Diff
    ret = ctx(bbcs.fn, args...)
    return ret, GenericCallSite(ctx.tr, ctx.score, bbcs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, bbcs::GenericCallSite) where L <: ConstrainedSelection
    argdiffs = NoChange()
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, bbcs.args...)
end

function update(sel::L, bbcs::GenericCallSite, argdiffs::D, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, vcs::VectorizedSite, argdiffs::D, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(vcs, sel, argdiffs)
    return update(ctx, vcs, new_args...)
end

function update(sel::L, vcs::VectorizedSite{typeof(plate)}) where {L <: ConstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, argdiffs)
    ret = ctx(plate, addr, vcs.kernel, vcs.args)
    return ret, VectorizedSite{typeof(plate)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end
