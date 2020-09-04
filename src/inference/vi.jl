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
    _, gs_est = get_learnable_gradients(ps, cl, 1.0, lw * scale)
    return gs_est, lw
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
    gs = Gradients()
    for s in 1 : gs_samples
        gs_est, lw = one_shot_gradient_estimator(tg, ps, v_mod, v_args, mod, args; scale = 1.0 / gs_samples)
        elbo_est += lw / gs_samples
        accumulate!(gs, gs_est)
    end
    ps = update_learnables(opt, ps, gs)
    ps, elbo_est
end

# ------------  ADVI with geometric baseline ------------ #

# TODO: finish testing.

function geometric_base(lws::Vector{Float64})
    est_samples = length(lws)
    s = sum(lws)
    baselines = Vector{Float64}(undef, est_samples)
    for i = 1 : est_samples
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
    for i in 1 : est_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        merge!(cs[i], tg) && error("(vimco_gradient_estimator): variational model proposes to addresses in observations.")
        ret, mlw = score(cs[i], ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    nw = exp.(lws .- ltw)
    L = ltw - log(est_samples)
    bs = geometric_base(lws)
    gs_est = Gradients()
    for i in 1 : est_samples
        ls = L - nw[i] - bs[i]
        _, gs = get_learnable_gradients(ps, cs[i], 1.0, ls * scale)
        accumulate!(gs_est, gs)
    end
    return gs_est, L
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
    gs = Gradients()
    for s in 1 : gs_samples
        gs_est, L = vimco_gradient_estimator(tg, ps, v_mod, v_args, mod, args; est_samples = est_samples, scale = 1.0 / gs_samples)
        velbo_est += L / gs_samples
        accumulate!(gs, gs_est)
    end
    ps = update_learnables(opt, ps, gs)
    ps, velbo_est
end
