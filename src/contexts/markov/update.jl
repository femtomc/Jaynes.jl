# ------------ Utilities ------------ #

function trace_retained(vcs::VectorCallSite, 
                        s::K, 
                        ps::P,
                        ks::Set, 
                        min::Int, 
                        o_len::Int, 
                        n_len::Int, 
                        args...) where {K <: ConstrainedSelection, P <: Parameters}
    w_adj = -sum(map(vcs.trace.subrecords[n_len + 1 : end]) do cl
                     get_score(cl)
                 end)
    new = vcs.trace.subrecords[1 : min - 1]
    new_ret = vcs.ret[1 : min - 1]

    # Start at min
    ss = get_sub(s, min)
    prev_cl = get_sub(vcs, min)
    if min == 1
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), args...)
    else
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), new_ret[min - 1]...)
    end
    push!(new_ret, ret)
    push!(new, u_cl)
    w_adj += u_w

    for i in min + 1 : n_len
        ss = get_sub(s, i)
        prev_cl = get_sub(vcs, i)
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), new_ret[i - 1]...)
        push!(new_ret, ret)
        push!(new, u_cl)
        w_adj += u_w
    end
    return w_adj, new, new_ret
end

# TODO: finish.
function trace_new(vcs::VectorCallSite, 
                   s::K, 
                   ps::P,
                   ks::Set, 
                   min::Int, 
                   o_len::Int, 
                   n_len::Int, 
                   args...) where {K <: ConstrainedSelection, P <: Parameters}
    w_adj = 0.0
    new = vcs.trace.subrecords[1 : min - 1]
    new_ret = vcs.ret[1 : min - 1]

    # Start at min, check if it's less than old length. Otherwise, constraints will be applied during generate.
    if min <= o_len
        ss = get_sub(s, min)
        prev_cl = get_sub(vcs, min)
        if min == 1
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), args...)
        else
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), new_ret[min - 1]...)
        end
        push!(new_ret, ret)
        push!(new, u_cl)
        w_adj += u_w

        # From min, apply constraints and compute updates to weight.
        for i in min + 1 : o_len
            ss = get_sub(s, i)
            prev_cl = get_sub(vcs, i)
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl, UndefinedChange(), new_ret[i - 1]...)
            push!(new_ret, ret)
            push!(new, u_cl)
            w_adj += u_w
        end
    end
    
    # Now, generate new call sites with constraints.
    for i in o_len + 1 : n_len
        ss = get_sub(s, i)
        ret, g_cl, g_w = generate(ss, ps, vcs.fn, new_ret[i - 1]...)
        push!(new_ret, ret)
        push!(new, g_cl)
        w_adj += g_w
    end

    return w_adj, new, new_ret
end

# TODO: finish.
function trace_new(vcs::VectorCallSite, 
                   s::ConstrainedEmptySelection, 
                   ps::P,
                   ks::Set, 
                   min::Int, 
                   o_len::Int, 
                   n_len::Int, 
                   args...) where P <: Parameters

    w_adj = 0.0
    new = vcs.trace.subrecords[1 : min - 1]
    new_ret = vcs.ret[1 : min - 1]

    for i in o_len + 1 : n_len
        ret, g_cl = simulate(ps, vcs.fn, new_ret[i - 1]...)
        push!(new_ret, ret)
        push!(new, g_cl)
    end

    return w_adj, new, new_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(markov), 
                                      addr::Address, 
                                      call::Function, 
                                      len::Int,
                                      args...)
    visit!(ctx, addr)
    vcs = get_prev(ctx, addr)
    n_len, o_len = len, length(vcs.ret)
    ps = get_sub(ctx.params, addr)
    s = get_sub(ctx.target, addr)
    min, ks = keyset(s, n_len)
    if n_len <= o_len
        w_adj, new, new_ret = trace_retained(vcs, s, ps, ks, min, o_len, n_len, args...)
    else
        w_adj, new, new_ret = trace_new(vcs, s, ps, ks, min, o_len, n_len, args...)
    end
    add_call!(ctx, addr, VectorCallSite{typeof(markov)}(VectorTrace(new), get_score(vcs) + w_adj, call, n_len, args, new_ret))
    increment!(ctx, w_adj)

    return new_ret
end

@inline function (ctx::UpdateContext{C, T})(c::typeof(markov), 
                                            call::Function, 
                                            len::Int,
                                            args...) where {C <: VectorCallSite, T <: VectorTrace}
    vcs = ctx.prev
    n_len, o_len = len, length(vcs.ret)
    ps = ctx.fixed
    s = ctx.select
    min, ks = keyset(s, n_len)
    if n_len <= o_len
        w_adj, new, new_ret = trace_retained(vcs, s, ps, ks, min, o_len, n_len, args...)
    else
        w_adj, new, new_ret = trace_new(vcs, s, ps, ks, min, o_len, n_len, args...)
    end

    # TODO: fix - allocate full vector.
    for n in new
        add_call!(ctx, n)
    end
    increment!(ctx, w_adj)

    return new_ret
end

# ------------ Convenience ------------ #

function update(sel::L, vcs::VectorCallSite{typeof(markov)}) where L <: AddressMap
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, argdiffs)
    ret = ctx(markov, vcs.fn, vcs.args[1], vcs.args[2]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorCallSite{typeof(markov)}) where {L <: AddressMap, P <: AddressMap}
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, ps, argdiffs)
    ret = ctx(markov, vcs.fn, vcs.args[1], vcs.args[2]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, D <: Diff}
    ctx = UpdateContext(vcs, sel, NoChange())
    ret = ctx(markov, vcs.fn, len, vcs.args[2]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, P <: AddressMap, D <: Diff}
    ctx = UpdateContext(vcs, sel, ps, NoChange())
    ret = ctx(markov, vcs.fn, len, vcs.args[2]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end
