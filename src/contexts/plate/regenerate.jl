# ------------ Utilities ------------ #

function trace_retained(vcs::VectorCallSite, 
                        s::K, 
                        ps,
                        ks, 
                        o_len::Int, 
                        n_len::Int, 
                        args::Vector) where K <: Target
    w_adj = 0.0
    new = get_choices(get_trace(vcs))[1 : n_len]
    new_ret = typeof(get_ret(vcs))(undef, n_len)
    for i in 1 : n_len
        if i in ks
            ss = get_sub(s, i)
            prev_cl = get_sub(vcs, i)
            ret, u_cl, u_w, rd, ds = regenerate(ss, ps, prev_cl)
            new_ret[i] = ret
            new[i] = u_cl
            w_adj += get_score(u_cl) - get_score(prev_cl)
        else
            new_ret[i] = get_ret(get_sub(vcs, i))
        end
    end
    return w_adj, new, new_ret
end

function trace_new(vcs::VectorCallSite, 
                   s::K, 
                   ps,
                   ks, 
                   o_len::Int, 
                   n_len::Int, 
                   args::Vector) where K <: Target
    w_adj = 0.0
    new_ret = typeof(get_ret(vcs))(undef, n_len)
    new = get_choices(vcs)
    for i in o_len + 1 : n_len
        ss = get_sub(s, i)
        ret, cl = simulate(call, args[i]...)
        new_ret[i] = ret
        new[i] = cl
    end
    for i in 1 : o_len
        i in ks && begin
            ss = get_sub(s, i)
            prev_cl = get_sub(vcs, i)
            ret, u_cl, u_w, rd, d = regenerate(ss, ps, prev_cl)
            new_ret[i] = ret
            new[i] = u_cl
            w_adj += u_w
            continue
        end
    end
    return w_adj, new, new_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::RegenerateContext{C, T})(c::typeof(plate), 
                                                call::Function, 
                                                args::Vector) where {C <: VectorCallSite, T <: VectorTrace}
    vcs = ctx.prev
    n_len, o_len = length(args), length(vcs.args)
    s = ctx.target
    ps = ctx.params
    _, ks = keyset(s, n_len)
    if n_len <= o_len
        w_adj, new, new_ret = trace_retained(vcs, s, ps, ks, o_len, n_len, args)
    else
        w_adj, new, new_ret = trace_new(vcs, s, ps, ks, o_len, n_len, args)
    end

    for (i, cl) in enumerate(new)
        add_call!(ctx, i, cl)
    end
    increment!(ctx, w_adj)

    return new_ret
end

@inline function (ctx::RegenerateContext)(c::typeof(plate), 
                                          addr::A, 
                                          call::Function, 
                                          args::Vector) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(get_trace(ctx.prev), addr)
        prev_call = get_prev(ctx, addr)
        ret, cl, w, retdiff, d = regenerate(ss, ps, prev_call)
    else
        ret, cl, w = generate(ss, ps, plate, call, args)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Convenience ------------ #

function regenerate(sel::L, vcs::VectorCallSite{typeof(plate)}) where {L <: Target, D <: Diff}
    ctx = Regenerate(sel, Empty(), vcs, VectorTrace(vcs.len), VectorDiscard(), NoChange())
    ret = ctx(plate, vcs.fn, vcs.args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, vcs.len), ctx.weight, UnknownChange(), ctx.discard
end

function regenerate(sel::L, ps::P, vcs::VectorCallSite{typeof(plate)}) where {L <: Target, P <: AddressMap, D <: Diff}
    ctx = Regenerate(sel, ps, vcs, VectorTrace(vcs.len), VectorDiscard(), NoChange())
    ret = ctx(plate, vcs.fn, vcs.args)
    return ret, VectorCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret, vcs.len), ctx.weight, UnknownChange(), ctx.discard
end
