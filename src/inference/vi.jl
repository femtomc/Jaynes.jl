function one_shot_gradient_estimator(sel::K,
                                     v_mod::Function,
                                     v_args::Tuple,
                                     mod::Function,
                                     args::Tuple;
                                     params = LearnableParameters()) where K <: ConstrainedSelection
    _, cl = simulate(v_mod, v_args...; params = params)
    obs = merge(cl, sel)
    _, mlw = score(obs, mod, args...; params = params)
    lw = mlw - get_score(cl)
    gs = get_parameter_gradients(cl, nothing, lw)
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
