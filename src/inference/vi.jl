function one_shot_gradient_estimator(sel::K,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple) where K <: ConstrainedSelection
    cl = trace(v_mod, v_args...)
    merge!(sel, get_selection(cl))
    mod_log_w = score(sel, mod, args...)
    println(mod_log_w)
    lw = mod_log_w - get_score(cl)
    println(lw)
    param_grads = get_parameter_gradients(cl, 1.0, lw)
    return lw, param_grads
end
