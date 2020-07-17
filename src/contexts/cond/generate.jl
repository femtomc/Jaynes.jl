# ------------ Call sites ------------ #

@inline function (ctx::GenerateContext)(::typeof(cond), 
                                        addr::Address, 
                                        c::Function, 
                                        args::Tuple,
                                        a::Function,
                                        a_args::Tuple,
                                        b::Function,
                                        b_args::Tuple)
    visit!(ctx, addr)
    ss = get_subselection(ctx, addr => :C)
    ret, cl, w = generate(ss, c, args...)
    if ret
        ss = get_subselection(ctx, addr => :A)
        branch_ret, branch_cl, branch_w = generate(ss, a, a_args...)
        br_tr = BranchTrace(cl, branch_cl)
        add_call!(ctx, addr, ConditionalBranchCallSite(br_tr, get_score(branch_cl) + get_score(cl), c, args, ret, a, a_args, branch_ret))
    else
        ss = get_subselection(ctx, addr => :B)
        branch_ret, branch_cl, branch_w = generate(ss, b, b_args...)
        br_tr = BranchTrace(cl, branch_cl)
        add_call!(ctx, addr, ConditionalBranchCallSite(br_tr, get_score(branch_cl) + get_score(cl), c, args, ret, b, b_args, ret))
    end
end
