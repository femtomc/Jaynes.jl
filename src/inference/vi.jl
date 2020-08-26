function one_shot_gradient_estimator(tg::K,
                                     ps::P,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple;
                                     scale = 1.0) where {K <: AddressMap, P <: AddressMap}
    _, cl = simulate(ps, v_mod, v_args...)
    merge!(cl, tg) && error("(one_shot_gradient_estimator): variational model proposes to addresses in observations.")
    _, mlw = score(cl, ps, mod, args...)
    lw = mlw - get_score(cl)
    _, gs = get_learnable_gradients(ps, cl, 1.0, lw * scale)
    return gs, lw, cl
end

# ------------ Automatic differentiation variational inference ------------ #

function automatic_differentiation_variational_inference(tg::K,
                                                         ps::P,
                                                         v_mod::Function,
                                                         v_args::Tuple,
                                                         mod::Function,
                                                         args::Tuple;
                                                         opt = ADAM(0.05, (0.9, 0.8)),
                                                         iters = 1000, 
                                                         gs_samples = 100) where {K <: AddressMap, P <: AddressMap}
    cls = Vector{CallSite}(undef, gs_samples)
    elbows = Vector{Float64}(undef, iters)
    Threads.@threads for i in 1 : iters
        elbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, lw, cl = one_shot_gradient_estimator(tg, ps, v_mod, v_args, mod, args; scale = 1.0 / gs_samples)
            elbo_est += lw / gs_samples
            accumulate!(gs_est, gs)
            cls[s] = cl
        end
        elbows[i] = elbo_est
        ps = update_learnables(opt, ps, gs_est)
    end
    ps, elbows, cls
end

function geometric_base(lws::Vector{Float64})
    est_samples = length(lws)
    s = sum(lws)
    baselines = Vector{Float64}(undef, est_samples)
    for i=1:est_samples
        temp = lws[i]
        lws[i] = (s - lws[i]) / (est_samples - 1)
        baselines[i] = lse(lws) - log(est_samples)
        lws[i] = temp
    end
    baselines
end

function vimco_gradient_estimator(tg::K,
                                  ps::P,
                                  v_mod::Function,
                                  v_args::Tuple,
                                  mod::Function,
                                  args::Tuple;
                                  est_samples::Int = 100,
                                  scale = 1.0) where {K <: AddressMap, P <: AddressMap}
    cs = Vector{CallSite}(undef, est_samples)
    lws = Vector{Float64}(undef, est_samples)
    Threads.@threads for i in 1:est_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        merge!(cs[i], tg) && error("(vimco_gradient_estimator): variational model proposes to addresses in observations.")
        ret, mlw = score(cs[i], ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(est_samples)
    nw = exp.(lws .- ltw)
    gs = Gradients()
    bs = geometric_base(lws)
    Threads.@threads for i in 1:est_samples
        ls = L - nw[i] - bs[i]
        accumulate!(gs, get_learnable_gradients(ps, cs[i], 1.0, ls * scale)[2])
    end
    return gs, L, cs, nw
end

# ------------  ADVI with geometric baseline ------------ #

function automatic_differentiation_geometric_vimco(tg::K,
                                                   ps::P,
                                                   est_samples::Int,
                                                   v_mod::Function,
                                                   v_args::Tuple,
                                                   mod::Function,
                                                   args::Tuple;
                                                   opt = ADAM(0.05, (0.9, 0.8)),
                                                   iters = 1000, 
                                                   gs_samples = 100) where {K <: AddressMap, P <: AddressMap}
    cls = Vector{CallSite}(undef, gs_samples)
    velbows = Vector{Float64}(undef, iters)
    Threads.@threads for i in 1 : iters
        velbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, L, cs, nw = vimco_gradient_estimator(tg, ps, v_mod, v_args, mod, args; est_samples = est_samples, scale = 1.0 / gs_samples)
            velbo_est += L / gs_samples
            accumulate!(gs_est, gs)
            cls[s] = cs[rand(Categorical(nw))]
        end
        velbows[i] = velbo_est
        ps = update_learnables(opt, ps, gs_est)
    end
    ps, velbows, cls
end
