function maximum_likelihood_estimation(ps::P,
                                       cl::C; 
                                       distance = Euclidean(),
                                       max_ss = 1.0,
                                       min_ss = 1e-16,
                                       τ = 0.999) where {P <: AddressMap, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        _, sc = score(cl, ps, get_model(cl), get_args(cl)...)
        _, gs = get_learnable_gradients(ps, cl, 1.0; scaler = 1.0)
        vals = array(ps, Float64)
        grads = array(gs, Float64)
        new = (vals + step_size * grads) + (step_size * randn(length(vals)))
        new_params = target(ps, new)
        _, new_cl, _ = update(Empty(), new_params, cl)
        if get_score(new_cl) - sc >= 0 
            δ = evaluate(distance, new, vals)
            return (δ, new_params, true)
        end
        step_size = τ * step_size
    end
    return (0.0, ps, false)
end

# ForwardDiff version of MLE for single address site changes.
function maximum_likelihood_estimation(tg::T,
                                       ps::P,
                                       cl::C; 
                                       distance = Euclidean(),
                                       max_ss = 1.0,
                                       min_ss = 1e-16,
                                       τ = 0.999) where {T <: Tuple, P <: AddressMap, C <: CallSite}
    step_size = max_ss
    while step_size > min_ss
        _, sc = score(cl, ps, get_model(cl), get_args(cl)...)
        val, grad = get_learnable_gradient(tg, ps, cl)
        new = (val + step_size * grad) + (step_size * randn())
        new_params, _ = merge(ps, target(tg => new))
        _, new_cl, _ = update(Empty(), new_params, cl)
        if get_score(new_cl) - sc >= 0 
            δ = evaluate(distance, new, val)
            return (δ, new_params, true)
        end
        step_size = τ * step_size
    end
    return (0.0, ps, false)
end
