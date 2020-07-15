function one_shot_gradient_estimator(sel::K,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple;
                                     scale = 1.0,
                                     params = LearnableParameters()) where K <: ConstrainedSelection
    _, cl = simulate(v_mod, v_args...; params = params)
    obs = merge(cl, sel)
    _, mlw = score(obs, mod, args...; params = params)
    lw = mlw - get_score(cl)
    gs = get_parameter_gradients(cl, nothing, lw * scale)
    return gs, lw, cl
end

function multi_shot_gradient_estimator(sel::K,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       num_samples::Int = 5000,
                                       params = LearnableParameters()) where K <: ConstrainedSelection
    cs = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    for i in 1:num_samples
        _, cs[i] = simulate(v_mod, v_args...; params = params)
        obs = merge(cs[i], sel)
        ret, mlw = score(obs, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(num_samples)
    nw = exp.(lws .- ltw)
    gs = Gradients()
    for i in 1:num_samples
        ls = L - nw[i]
        accumulate_parameter_gradients!(gs, cs[i], nothing, ls)
    end
    return gs, L, cs, nw
end

function advi(sel::K,
              v_mod::Function,
              v_args::Tuple,
              mod::Function,
              args::Tuple;
              opt = ADAM(),
              iters = 1000, 
              gs_samples = 100) where K <: ConstrainedSelection
    cls = Vector{CallSite}(undef, gs_samples)
    elbows = Vector{Float64}(undef, iters)
    _, cl = simulate(v_mod, v_args...)
    params = get_parameters(cl)
    for i in 1 : iters
        elbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, lw, cl = one_shot_gradient_estimator(sel, v_mod, v_args, mod, args; scale = 1 / gs_samples, params = params)
            elbo_est += lw / gs_samples
            gs_est += gs
            cls[s] = cl
        end
        elbows[i] = elbo_est
        params = update_parameters(opt, params, gs_est)
    end
    params, elbows, cls
end
