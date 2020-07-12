function learnable_normal(x::Float64, y::Float64)
    l = learnable(:l, 3.0)
    m = learnable(:m, 10.0)
    q = rand(:q, Normal(l, 3 + m^2))
    return q
end

@testset "Convergence for learning - MAP 1" begin
    ret, cl, w = generate(selection((:q, 6.0)), learnable_normal, 5.0, 3.0)
    params = get_parameters(cl)
    for i in 1:100
        ret, cl, w = generate(selection((:q, 6.0)), learnable_normal, 5.0, 3.0; params = params)
        param_grads = get_parameter_gradients(cl, 1.0)
        update!(params, param_grads)
    end
    @test params.utility[:l] ≈ 6.0 atol = 1e-3
    @test params.utility[:m] ≈ 0.0 atol = 1e-3
end

function model()
    slope = rand(:slope, Normal(-1, exp(0.5)))
    intercept = rand(:intercept, Normal(1, exp(2.0)))
end

function var()
    slope_mu = learnable(:slope_mu, 0.0)
    slope_log_std = learnable(:slope_log_std, 0.0)
    intercept_mu = learnable(:intercept_mu, 0.0)
    intercept_log_std = learnable(:intercept_log_std, 0.0)
    slope = rand(:slope, Normal(slope_mu, exp(slope_log_std)))
    intercept = rand(:intercept, Normal(intercept_mu, exp(intercept_log_std)))
end

#@testset "Convergence for learning - VI 1" begin
#    sel = selection()
#    ret, cl, w = generate(sel, var)
#    params = get_parameters(cl)
#    for i in 1:100
#        _, cl = simulate(var; params = params)
#        obs = merge(cl, sel)
#        _, mlw = score(obs, model; params = params)
#        lw = mlw - get_score(cl)
#        gs = get_parameter_gradients(cl, nothing, lw)
#        update!(params, gs)
#    end
#    @test params.utility[:slope_mu] ≈ -1.0 atol = 1e-3
#    @test params.utility[:slope_log_std] ≈ 0.5 atol = 1e-3
#    @test params.utility[:intercept_mu] ≈ 1.0 atol = 1e-3
#    @test params.utility[:intercept_log_std] ≈ 2.0 atol = 1e-3
#end

#function learnable_hypers(x::Float64, y::Float64)
#    l = learnable(:l, 3.0)
#    m = learnable(:m, 10.0)
#    p = rand(:p, Normal(l, 1.0))
#    t = rand(:t, Normal(m, 1.0))
#    q = rand(:q, Normal(p + t, 1.0))
#    return q
#end
#
#@testset "Convergence for learnable hypers" begin
#    cl, w = generate(selection((:q, 6.0)), learnable_hypers, 5.0, 3.0)
#    params = get_parameters(cl)
#    for i in 1:1000
#        cl, w = generate(selection((:q, 6.0)), learnable_hypers, 5.0, 3.0; params = params)
#        param_grads = get_parameter_gradients(cl, 1.0)
#        update!(params, param_grads)
#        display(params; show_values = true)
#    end
#    @test params.utility[:l] ≈ 6.0 atol = 1e-3
#    @test params.utility[:m] ≈ 0.0 atol = 1e-3
#end
