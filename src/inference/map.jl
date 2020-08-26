function maximum_a_posteriori(tg::T, 
                              cl::C; 
                              max_ss = 1.0,
                              min_ss = 1e-16,
                              τ = 0.999) where {T <: Target, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        _, am, gs = get_choice_gradients(tg, cl, 1.0)
        vals = array(am, Float64)
        grads = array(gs, Float64)
        score = get_score(cl)
        new = (vals + step_size * grads) + (step_size * randn(length(vals)))
        new_map = target(am, new)
        _, new_cl, _ = update(new_map, cl)
        if get_score(new_cl) - score >= 0 
            return (new_cl, true)
        end
        step_size = τ * step_size
    end
    return (cl, false)
end
