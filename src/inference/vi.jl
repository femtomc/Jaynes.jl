function one_shot_gradient_estimator(sel::K,
                                     params::P,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple;
                                     scale = 1.0) where {K <: ConstrainedSelection, P <: Parameters}
    _, cl = simulate(params, v_mod, v_args...)
    obs = merge(cl, sel)
    _, mlw = score(obs, params, mod, args...)
    lw = mlw - get_score(cl)
    gs = get_learnable_gradients(params, cl, nothing, lw * scale)
    return gs, lw, cl
end

function multi_shot_gradient_estimator(sel::K,
                                       params::P,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       num_samples::Int = 5000) where {K <: ConstrainedSelection, P <: Parameters}
    cs = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    Threads.@threads for i in 1:num_samples
        _, cs[i] = simulate(params, v_mod, v_args...)
        obs = merge(cs[i], sel)
        ret, mlw = score(obs, params, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(num_samples)
    nw = exp.(lws .- ltw)
    gs = Gradients()
    Threads.@threads for i in 1:num_samples
        ls = L - nw[i]
        accumulate_parameter_gradients!(params, gs, cs[i], nothing, ls)
    end
    return gs, L, cs, nw
end

# ------------ Automatic differentiation variational inference ------------ #

function automatic_differentiation_variational_inference(sel::K,
                                                         params::P,
                                                         v_mod::Function,
                                                         v_args::Tuple,
                                                         mod::Function,
                                                         args::Tuple;
                                                         opt = ADAM(),
                                                         iters = 1000, 
                                                         gs_samples = 100) where {K <: ConstrainedSelection, P <: Parameters}
    cls = Vector{CallSite}(undef, gs_samples)
    elbows = Vector{Float64}(undef, iters)
    Threads.@threads for i in 1 : iters
        elbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, lw, cl = one_shot_gradient_estimator(sel, params, v_mod, v_args, mod, args; scale = 1 / gs_samples)
            elbo_est += lw / gs_samples
            gs_est += gs
            cls[s] = cl
        end
        elbows[i] = elbo_est
        params = update_learnables(opt, params, gs_est)
    end
    params, elbows, cls
end
