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

# ForwardDiff for single site address gradients.
function one_shot_gradient_estimator(addr::T,
                                     tg::K,
                                     ps::P,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple) where {T <: Tuple, K <: AddressMap, P <: AddressMap}
    _, cl = simulate(ps, v_mod, v_args...)
    merge!(cl, tg) && error("(one_shot_gradient_estimator): variational model proposes to addresses in observations.")
    _, mlw = score(cl, ps, mod, args...)
    lw = mlw - get_score(cl)
    _, gs_est = get_learnable_gradient(addr, ps, cl)
    return gs_est, lw
end

function automatic_differentiation_variational_inference(opt,
                                                         addr::T,
                                                         tg::K,
                                                         ps::P,
                                                         v_mod::Function,
                                                         v_args::Tuple,
                                                         mod::Function,
                                                         args::Tuple;
                                                         learning_rate = 0.01,
                                                         gs_samples = 100) where {T <: Tuple, K <: AddressMap, P <: AddressMap}
    elbo_est = 0.0
    gs = 0.0
    scale = 1.0 / gs_samples
    val = getindex(ps, addr)
    for s in 1 : gs_samples
        gs_est, lw = one_shot_gradient_estimator(addr, tg, ps, v_mod, v_args, mod, args)
        elbo_est += lw / gs_samples
        gs += gs_est * scale
    end
    new = val + learning_rate * gs
    ps = merge(ps, target(addr => new))
    ps, elbo_est
end

# ------------  ADVI with geometric baseline ------------ #

# TODO: finish testing.

function geometric_base(lws::Vector{Float64})
    n_samples = length(lws)
    s = sum(lws)
    baselines = Vector{Float64}(undef, n_samples)
    for i = 1 : n_samples
        temp = lws[i]
        lws[i] = (s - lws[i]) / (n_samples - 1)
        baselines[i] = lse(lws) - log(n_samples)
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
                                  n_samples::Int = 100) where {K <: AddressMap, P <: AddressMap}
    cs = Vector{CallSite}(undef, n_samples)
    lws = Vector{Float64}(undef, n_samples)
    for i in 1 : n_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        merge!(cs[i], tg) && error("(vimco_gradient_estimator): variational model proposes to addresses in observations.")
        _, mlw = score(cs[i], ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(n_samples)
    nw = exp.(lws .- ltw)
    bs = geometric_base(lws)
    gs_est = Gradients()
    for i in 1 : n_samples
        ls = L - nw[i] - bs[i]
        _, gs = get_learnable_gradients(ps, cs[i], 1.0, scale * ls / n_samples)
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
                                                   n_samples = 100,
                                                   gs_samples = 100) where {K <: AddressMap, P <: AddressMap}
    velbo_est = 0.0
    gs = Gradients()
    for s in 1 : gs_samples
        gs_est, L = vimco_gradient_estimator(tg, ps, 
                                             v_mod, v_args, 
                                             mod, args; 
                                             n_samples = n_samples, scale = 1.0 / gs_samples)
        velbo_est += L / gs_samples
        accumulate!(gs, gs_est)
    end
    ps = update_learnables(opt, ps, gs)
    ps, velbo_est
end

# ForwardDiff for single site address gradients.
function vimco_gradient_estimator(addr::T,
                                  tg::K,
                                  ps::P,
                                  v_mod::Function,
                                  v_args::Tuple,
                                  mod::Function,
                                  args::Tuple;
                                  n_samples::Int = 100) where {T <: Tuple, K <: AddressMap, P <: AddressMap}
    cs = Vector{CallSite}(undef, n_samples)
    lws = Vector{Float64}(undef, n_samples)
    for i in 1 : n_samples
        _, cs[i] = simulate(ps, v_mod, v_args...)
        merge!(cs[i], tg) && error("(vimco_gradient_estimator): variational model proposes to addresses in observations.")
        _, mlw = score(cs[i], ps, mod, args...)
        lws[i] = mlw - get_score(cs[i])
    end
    ltw = lse(lws)
    L = ltw - log(n_samples)
    nw = exp.(lws .- ltw)
    bs = geometric_base(lws)
    gs_est = 0.0
    for i in 1 : n_samples
        ls = L - nw[i] - bs[i]
        _, gs = get_learnable_gradient(addr, ps, cs[i])
        gs_est += gs * (ls / n_samples)
    end
    return gs_est, L
end

function automatic_differentiation_geometric_vimco(opt,
                                                   addr::T,
                                                   tg::K,
                                                   ps::P,
                                                   v_mod::Function,
                                                   v_args::Tuple,
                                                   mod::Function,
                                                   args::Tuple;
                                                   n_samples = 100,
                                                   gs_samples = 100) where {T <: Tuple, K <: AddressMap, P <: AddressMap}
    velbo_est = 0.0
    gs = 0.0
    val = getindex(ps, addr)
    scale = 1.0 / gs_samples
    for s in 1 : gs_samples
        gs_est, L = vimco_gradient_estimator(addr, tg, ps, 
                                             v_mod, v_args, 
                                             mod, args; 
                                             n_samples = n_samples)
        velbo_est += L * scale
        gs += gs_est * scale
    end
    new = val + learning_rate * gs
    ps = merge(ps, target(addr => new))
    ps, velbo_est
end
