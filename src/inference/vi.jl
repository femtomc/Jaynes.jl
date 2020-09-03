# ------------ Automatic differentiation variational inference ------------ #

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
    return gs, lw
end

function automatic_differentiation_variational_inference(opt,
                                                         tg::K,
                                                         ps::P,
                                                         v_mod::Function,
                                                         v_args::Tuple,
                                                         mod::Function,
                                                         args::Tuple;
                                                         gs_samples = 100) where {K <: AddressMap, P <: AddressMap}
    elbo_est = 0.0
    gs_est = Gradients()
    for s in 1 : gs_samples
        gs, lw = one_shot_gradient_estimator(tg, ps, v_mod, v_args, mod, args; scale = 1.0 / gs_samples)
        elbo_est += lw / gs_samples
        accumulate!(gs_est, gs)
    end
    ps = update_learnables(opt, ps, gs_est)
    ps, elbo_est
end

# ------------  ADVI with geometric baseline ------------ #

# TODO: finish testing.

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
        gs_est, _ = get_learnable_gradients(ps, cs[i], 1.0, ls * scale)
        accumulate!(gs, gs_est)
    end
    return gs, L
end

function automatic_differentiation_geometric_vimco(opt,
                                                   tg::K,
                                                   ps::P,
                                                   v_mod::Function,
                                                   v_args::Tuple,
                                                   mod::Function,
                                                   args::Tuple;
                                                   est_samples = 100,
                                                   gs_samples = 100) where {K <: AddressMap, P <: AddressMap}
    velbo_est = 0.0
    gs_est = Gradients()
    for s in 1 : gs_samples
        gs, L = vimco_gradient_estimator(tg, ps, v_mod, v_args, mod, args; est_samples = est_samples, scale = 1.0 / gs_samples)
        velbo_est += L / gs_samples
        accumulate!(gs_est, gs)
    end
    ps = update_learnables(opt, ps, gs_est)
    ps, velbo_est
end
