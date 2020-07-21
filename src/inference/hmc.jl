function hamiltonian_monte_carlo(sel::K, cl::C; L=10, eps=0.1) where {K <: UnconstrainedSelection, C <: CallSite}
    pr_score = get_score(cl)
    sel_values, choice_grads = get_choice_gradients(cl, sel, 1.0)
    vals = array(sel_values, Float64)
    grads = array(choice_grads, Float64)
    mv = MvNormal(length(values), 1.0)
    mom = rand(MvNormal(length(values), 1.0))
    p_mom_score = logpdf(mv, mom)
    for step=1:L
        mom += (eps / 2) * grads
        vals += eps * mom
        sel_values = from_array(vals, values)
        ret, u_cl, w, _ = update(sel_values, cl)
        _, choice_grads = get_choice_gradients(u_cl, sel, retval_grad)
        grads = array(choice_grads, Float64)
        mom += (eps / 2) * grads
    end
    n_mod_score = get_score(u_cl)
    n_mom_score = logpdf(d, -mom)
    alpha = get_score(u_cl) - p_mod_score + n_mom_score - p_mom_score

    # Accept or reject.
    if log(rand()) < alpha
        (u_cl, true)
    else
        (cl, false)
    end
end
