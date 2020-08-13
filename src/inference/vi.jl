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
    _, gs = get_learnable_gradients(ps, cl, nothing, lw * scale)
    return gs, lw, cl
end

# ------------ Automatic differentiation variational inference ------------ #

function automatic_differentiation_variational_inference(sel::K,
                                                         ps::P,
                                                         v_mod::Function,
                                                         v_args::Tuple,
                                                         mod::Function,
                                                         args::Tuple;
                                                         opt = ADAM(0.05, (0.9, 0.8)),
                                                         iters = 1000, 
                                                         gs_samples = 100) where {K <: ConstrainedSelection, P <: Parameters}
    cls = Vector{CallSite}(undef, gs_samples)
    elbows = Vector{Float64}(undef, iters)
    Threads.@threads for i in 1 : iters
        elbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, lw, cl = one_shot_gradient_estimator(sel, ps, v_mod, v_args, mod, args; scale = 1.0 / gs_samples)
            elbo_est += lw / gs_samples
            gs_est += gs
            cls[s] = cl
        end
        elbows[i] = elbo_est
        ps = update_learnables(opt, ps, gs_est)
    end
    ps, elbows, cls
end

function geometric_base(lws::Vector{Float64})
    num_samples = length(lws)
    s = sum(lws)
    baselines = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        temp = lws[i]
        lws[i] = (s - lws[i]) / (num_samples - 1)
        baselines[i] = lse(lws) - log(num_samples)
        lws[i] = temp
    end
    baselines
end

function lde(x, y)
    m = max(x, y)
    m + log(exp(x - m) - exp(y - m))
end

function multi_shot_gradient_estimator(sel::K,
                                       ps::P,
                                       v_mod::Function,
                                       v_args::Tuple,
                                       mod::Function,
                                       args::Tuple;
                                       num_samples::Int = 100,
                                       scale = 1.0) where {K <: ConstrainedSelection, P <: Parameters}
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
    bs = geometric_base(lws)
    Threads.@threads for i in 1:num_samples
        ls = L - nw[i] - bs[i]
        gs += get_learnable_gradients(ps, cs[i], nothing, ls * scale)[2]
    end
    return gs, L, cs, nw
end

# ------------  ADVI with geometric baseline ------------ #

function automatic_differentiation_geometric_vimco(sel::K,
                                                   ps::P,
                                                   num_samples::Int,
                                                   v_mod::Function,
                                                   v_args::Tuple,
                                                   mod::Function,
                                                   args::Tuple;
                                                   opt = ADAM(0.05, (0.9, 0.8)),
                                                   iters = 1000, 
                                                   gs_samples = 100) where {K <: ConstrainedSelection, P <: Parameters}
    cls = Vector{CallSite}(undef, gs_samples)
    velbows = Vector{Float64}(undef, iters)
    Threads.@threads for i in 1 : iters
        velbo_est = 0.0
        gs_est = Gradients()
        for s in 1 : gs_samples
            gs, L, cs, nw = multi_shot_gradient_estimator(sel, ps, v_mod, v_args, mod, args; num_samples = num_samples, scale = 1.0 / gs_samples)
            velbo_est += L / gs_samples
            gs_est += gs
            cls[s] = cs[rand(Categorical(nw))]
        end
        velbows[i] = velbo_est
        ps = update_learnables(opt, ps, gs_est)
    end
    ps, velbows, cls
end
