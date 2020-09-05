function maximum_a_posteriori_estimation(tg::T, 
                                         cl::C; 
                                         distance = Euclidean(),
                                         max_ss = 1.0,
                                         min_ss = 1e-16,
                                         τ = 0.999) where {T <: Target, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        am, _, gs = get_choice_gradients(tg, cl, 1.0)
        vals = array(am, Float64)
        grads = array(gs, Float64)
        score = get_score(cl)
        new = (vals + step_size * grads) + (step_size * randn(length(vals)))
        new_map = target(am, new)
        _, new_cl, _ = update(new_map, cl)
        if get_score(new_cl) - score >= 0 
            δ = evaluate(distance, new, vals)
            return (δ, new_cl, true)
        end
        step_size = τ * step_size
    end
    return (0.0, cl, false)
end

function maximum_a_posteriori_estimation(tg::T, 
                                         ps::P,
                                         cl::C; 
                                         distance = Euclidean(),
                                         max_ss = 1.0,
                                         min_ss = 1e-16,
                                         τ = 0.999) where {T <: Target, P <: AddressMap, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        am, _, gs = get_choice_gradients(tg, ps, cl, 1.0)
        vals = array(am, Float64)
        grads = array(gs, Float64)
        score = get_score(cl)
        new = (vals + step_size * grads) + (step_size * randn(length(vals)))
        new_map = target(am, new)
        _, new_cl, _ = update(new_map, ps, cl)
        if get_score(new_cl) - score >= 0 
            δ = evaluate(distance, new, vals)
            return (δ, new_cl, true)
        end
        step_size = τ * step_size
    end
    return (0.0, cl, false)
end

# ForwardDiff version of MAP for single address site changes.
function maximum_a_posteriori_estimation(tg::T, 
                                         cl::C; 
                                         distance = Euclidean(),
                                         max_ss = 1.0,
                                         min_ss = 1e-16,
                                         τ = 0.999) where {T <: Tuple, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        val, grad = get_choice_gradient(tg, cl)
        score = get_score(cl)
        new = (val + step_size * grad) + (step_size * randn())
        new_map = target(tg => new)
        _, new_cl, _ = update(new_map, cl)
        if get_score(new_cl) - score >= 0 
            δ = evaluate(distance, new, val)
            return (δ, new_cl, true)
        end
        step_size = τ * step_size
    end
    return (0.0, cl, false)
end

function maximum_a_posteriori_estimation(tg::T, 
                                         ps::P,
                                         cl::C; 
                                         distance = Euclidean(),
                                         max_ss = 1.0,
                                         min_ss = 1e-16,
                                         τ = 0.999) where {T <: Tuple, P <: AddressMap, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        val, grad = get_choice_gradient(tg, ps, cl)
        score = get_score(cl)
        new = (val + step_size * grad) + (step_size * randn())
        new_map = target(tg => new)
        _, new_cl, _ = update(new_map, ps, cl)
        if get_score(new_cl) - score >= 0 
            δ = evaluate(distance, new, val)
            return (δ, new_cl, true)
        end
        step_size = τ * step_size
    end
    return (0.0, cl, false)
end
