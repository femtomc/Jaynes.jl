# ------------ Call sites ------------ #

@inline function (ctx::SimulateContext)(::typeof(cond), 
                                        addr::T, 
                                        c::Function, 
                                        args::Tuple,
                                        a::Function,
                                        a_args::Tuple,
                                        b::Function,
                                        b_args::Tuple) where T <: Address
    visit!(ctx, addr)
    ret, cl = simulate(c, args...)
    if ret
        branch_ret, branch_cl = simulate(a, a_args...)
        br_tr = BranchTrace(cl, branch_cl)
        add_call!(ctx, addr, ConditionalBranchSite(br_tr, get_score(branch_cl) + get_score(cl), c, args, ret, a, a_args, branch_ret))
    else
        branch_ret, branch_cl = simulate(b, b_args...)
        br_tr = BranchTrace(cl, branch_cl)
        add_call!(ctx, addr, ConditionalBranchSite(br_tr, get_score(branch_cl) + get_score(cl), c, args, ret, b, b_args, ret))
    end
end

