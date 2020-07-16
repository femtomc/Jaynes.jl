# ------------ Utilities ------------ #

function trace_retained(vcs::VectorizedSite, s::ConstrainedSelection, ks, o_len::Int, n_len::Int, args::Vector)
    sc_adj = -sum(map(vcs.trace.subrecords[n_len + 1 : end]) do cl
                      get_score(cl)
                  end)
    w_adj = 0.0
    new = vcs.trace.subrecords[1 : n_len]
    new_ret = typeof(vcs.ret)(undef, n_len)
    for i in 1 : n_len
        i in ks && begin
            ss = get_sub(s, i)
            ret, u_cl, u_w, rd, d = update(ss, get_call(vcs, i), UndefinedChange(), args[i]...)
            new_ret[i] = ret
            new[i] = u_cl
            sc_adj += get_score(u_cl) - get_score(get_call(vcs, i))
            w_adj += u_w
        end
    end
    return w_adj, sc_adj, new, new_ret
end

function trace_new(vcs::VectorizedSite, s::ConstrainedSelection, ks, o_len::Int, n_len::Int, args::Vector)
    sc_adj = 0.0
    w_adj = 0.0
    new_ret = typeof(vcs.ret)(undef, n_len)
    new = vcs.trace.subrecords
    for i in o_len : n_len - 1
        ss = get_sub(s, i)
        ret, cl, w = generate(ss, call, args[i]...)
        new_ret[i] = ret
        new[i] = cl
        sc_adj += get_score(cl)
        w_adj += w
    end
    for i in 1 : o_len
        i in ks && begin
            ss = get_sub(s, i)
            ret, u_cl, u_w, rd, d = update(ss, get_call(vcs, i), UndefinedChange(), args[i]...)
            new_ret[i] = ret
            new[i] = u_cl
            sc_adj += get_score(u_cl) - get_score(get_call(vcs, i))
            w_adj += u_w
            continue
        end
    end
    return w_adj, sc_adj, new, new_ret
end

# ------------ Call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(plate), 
                                      addr::Address, 
                                      call::Function, 
                                      args::Vector)
    local sc_adj::Float64

    # TODO: fix with dispatch.
    if has_call(ctx.prev.trace, addr) 
        vcs = get_prev(ctx, addr)
    else
        vcs = ctx.prev
    end

    n_len, o_len = length(args), length(vcs.args)
    s = get_subselection(ctx, addr)
    ks = keyset(s, n_len)
    if n_len <= o_len
        w_adj, sc_adj, new, new_ret = trace_retained(vcs, s, ks, o_len, n_len, args)
    else
        w_adj, sc_adj, new, new_ret = trace_new(vcs, s, ks, o_len, n_len, args)
    end

    # TODO: fix with dispatch.
    for n in new
        add_call!(ctx, n)
    end
    increment!(ctx, w_adj)

    return new_ret
end
