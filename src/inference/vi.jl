function one_shot_gradient_estimator(sel::K,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple) where K <: ConstrainedSelection
    # Generate from variational model.
    _, cl = trace(v_mod, v_args...)

    # Get sample, merge into observation interfaces.
    merge!(sel, get_selection(cl))

    # Compute the score of the variational sample with respect to the original model.
    _, mlw = score(sel, mod, args...)

    # Compute the likelihood weight.
    lw = mlw - get_score(cl)

    # Compute the gradients with respect to the learnable parameters, scale them by the likelihood weight.
    param_grads = get_parameter_gradients(cl, 1.0, lw)

    # Return the likelihood weight and the scales gradients for all learnable parameters.
    return lw, param_grads
end
