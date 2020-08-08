function one_shot_gradient_estimator(sel::K,
                                     ps::P,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple;
                                     scale = 1.0) where {K <: ConstrainedSelection, P <: Parameters}
    _, cl = simulate(ps, v_mod, v_args...)
    obs, _ = merge(cl, sel)
    _, mlw = score(obs, ps, mod, args...)
    lw = mlw - get_score(cl)
    gs = get_learnable_gradients(ps, cl, nothing, lw * scale)
    return gs, lw, cl
end

function multi_shot_gradient_estimator(sel::K,
                                       ps::P,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       num_samples::Int = 5000) where {K <: ConstrainedSelection, P <: Parameters}
    cs = Vector{CallSite}(undef, num_samples)
    lws = Vector{Float64}(undef, num_samples)
    Threads.@threads for i in 1:num_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        obs, _ = merge(cs[i], sel)
        ret, mlw = score(obs, ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(num_samples)
    nw = exp.(lws .- ltw)
    gs = Gradients()
    for i in 1:num_samples
        ls = L - nw[i]
        accumulate_learnable_gradients!(ps, gs, cs[i], nothing, ls)
    end
    return gs, L, cs, nw
end

# ------------ Automatic differentiation variational inference ------------ #

function automatic_differentiation_variational_inference(sel::K,
                                                         ps::P,
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
            gs, lw, cl = one_shot_gradient_estimator(sel, ps, v_mod, v_args, mod, args; scale = 1 / gs_samples)
            elbo_est += lw / gs_samples
            gs_est += gs
            cls[s] = cl
        end
        elbows[i] = elbo_est
        ps = update_learnables(opt, ps, gs_est)
    end
    ps, elbows, cls
end
