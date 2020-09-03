function hamiltonian_monte_carlo(sel::K, 
                                 cl::C; 
                                 L=10, eps=0.1) where {K <: Target, C <: CallSite}
    local u_cl = cl
    p_mod_score = get_score(u_cl)
    _, sel_values, choice_grads = get_choice_gradients(sel, u_cl, 1.0)
    vals = array(sel_values, Float64)
    grads = array(choice_grads, Float64)
    d = MvNormal(length(vals), 1.0)
    mom = rand(d)
    p_mom_score = logpdf(d, mom)
    for step in 1 : L
        mom += (eps / 2) * grads
        vals += eps * mom
        sel_values = target(sel_values, vals)
        ret, u_cl, w, _ = update(sel_values, u_cl)
        _, _, choice_grads = get_choice_gradients(sel, u_cl, 1.0)
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

function hamiltonian_monte_carlo(sel::K, 
                                 ps::P,
                                 cl::C; 
                                 L=10, eps=0.1) where {K <: Target, P <: AddressMap, C <: CallSite}
    local u_cl = cl
    p_mod_score = get_score(u_cl)
    _, sel_values, choice_grads = get_choice_gradients(sel, u_cl, 1.0)
    vals = array(sel_values, Float64)
    grads = array(choice_grads, Float64)
    d = MvNormal(length(vals), 1.0)
    mom = rand(d)
    p_mom_score = logpdf(d, mom)
    for step in 1 : L
        mom += (eps / 2) * grads
        vals += eps * mom
        sel_values = target(sel_values, vals)
        ret, u_cl, w, _ = update(sel_values, ps, u_cl)
        _, _, choice_grads = get_choice_gradients(sel, ps, u_cl, 1.0)
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

# ForwardDiff version of HMC for single address site changes.
function hamiltonian_monte_carlo(sel::K, 
                                 cl::C; 
                                 L=10, eps=0.1) where {K <: Tuple, C <: CallSite}
    local u_cl = cl
    p_mod_score = get_score(u_cl)
    val, grad = get_choice_gradient(sel, u_cl)
    d = Normal(0.0, 1.0)
    mom = rand(d)
    p_mom_score = logpdf(d, mom)
    for step in 1 : L
        mom += (eps / 2) * grad
        val += eps * mom
        sel_values = target([sel => val])
        ret, u_cl, w, _ = update(sel_values, u_cl)
        _, grad = get_choice_gradient(sel, u_cl)
        mom += (eps / 2) * grad
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
