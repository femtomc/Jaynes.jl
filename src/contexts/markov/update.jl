# ------------ Utilities ------------ #

function trace_retained(vcs::VectorCallSite, 
                        s::K, 
                        ps::P,
                        ks::Set, 
                        min::Int, 
                        o_len::Int, 
                        n_len::Int, 
                        args...) where {K <: AddressMap, P <: AddressMap}
    w_adj = 0.0
    new = get_choices(vcs)[1 : min - 1]
    new_ret = get_ret(vcs)[1 : min - 1]

    # Start at min
    ss = get_sub(s, min)
    prev_cl = get_sub(vcs, min)
    if min == 1
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
    else
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
    end
    push!(new_ret, ret)
    push!(new, u_cl)
    w_adj += u_w

    for i in min + 1 : n_len
        ss = get_sub(s, i)
        prev_cl = get_sub(vcs, i)
        ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
        push!(new_ret, ret)
        push!(new, u_cl)
        w_adj += u_w
    end

    return w_adj, new, new_ret
end

function trace_new(vcs::VectorCallSite, 
                   s::K, 
                   ps::P,
                   ks::Set, 
                   min::Int, 
                   o_len::Int, 
                   n_len::Int, 
                   args...) where {K <: AddressMap, P <: AddressMap}
    w_adj = 0.0
    new = get_choices(get_trace(vcs))[1 : min - 1]
    new_ret = vcs.ret[1 : min - 1]

    # Start at min, check if it's less than old length. Otherwise, constraints will be applied during generate.
    if min <= o_len
        ss = get_sub(s, min)
        prev_cl = get_sub(vcs, min)
        if min == 1
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
        else
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
        end
        push!(new_ret, ret)
        push!(new, u_cl)
        w_adj += u_w

        # From min, apply constraints and compute updates to weight.
        for i in min + 1 : o_len
            ss = get_sub(s, i)
            prev_cl = get_sub(vcs, i)
            ret, u_cl, u_w, rd, ds = update(ss, ps, prev_cl)
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

function trace_new(vcs::VectorCallSite, 
                   s::Empty, 
                   ps::P,
                   ks::Set, 
                   min::Int, 
                   o_len::Int, 
                   n_len::Int, 
                   args...) where P <: AddressMap

    w_adj = 0.0
    new = get_choices(get_trace(vcs))[1 : min - 1]
    new_ret = vcs.ret[1 : min - 1]

    for i in o_len + 1 : n_len
        ret, g_cl = simulate(ps, vcs.fn, new_ret[i - 1]...)
        push!(new_ret, ret)
        push!(new, g_cl)
    end

    return w_adj, new, new_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::UpdateContext{C, T})(c::typeof(markov), 
                                            call::Function, 
                                            len::Int,
                                            args...) where {C <: VectorCallSite, T <: VectorTrace}
    vcs = ctx.prev
    n_len, o_len = len, length(vcs.ret)
    ps = ctx.params
    s = ctx.target
    min, ks = keyset(s, n_len)
    
    if n_len <= o_len
        w_adj, new, new_ret = trace_retained(vcs, s, ps, ks, min, o_len, n_len, args...)
    else
        w_adj, new, new_ret = trace_new(vcs, s, ps, ks, min, o_len, n_len, args...)
    end

    # Mutates the trace.
    for (i, cl) in enumerate(new)
        add_call!(ctx, i, cl)
    end
    increment!(ctx, w_adj)

    return new_ret
end

@inline function (ctx::UpdateContext)(c::typeof(markov), 
                                      addr::A, 
                                      call::Function, 
                                      len::Int,
                                      args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(ctx.prev, addr)
        prev = get_sub(ctx.prev, addr)
        ret, cl, w, rd, d = update(ss, ps, prev)
    else
        ret, cl, w = generate(ss, ps, markov, call, len, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function update(sel::L, vcs::VectorCallSite{typeof(markov)}) where L <: AddressMap
    ctx = Update(sel, Empty(), vcs, VectorTrace(vcs.len), VectorDiscard(), NoChange())
    ret = ctx(markov, vcs.fn, vcs.len, vcs.args[1]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, vcs.len), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorCallSite{typeof(markov)}) where {L <: AddressMap, P <: AddressMap}
    ctx = Update(sel, ps, vcs, VectorTrace(vcs.len), VectorDiscard(), NoChange())
    ret = ctx(markov, vcs.fn, vcs.len, vcs.args[1]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, vcs.len), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, D <: Diff}
    ctx = Update(sel, Empty(), vcs, VectorTrace(len), VectorDiscard(), NoChange())
    ret = ctx(markov, vcs.fn, vcs.len, vcs.args[1]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, len), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, P <: AddressMap, D <: Diff}
    ctx = Update(sel, ps, vcs, VectorTrace(len), VectorDiscard(), NoChange())
    ret = ctx(markov, vcs.fn, vcs.len, vcs.args[1]...)
    return ret, VectorCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, len), ctx.weight, UndefinedChange(), ctx.discard
end
