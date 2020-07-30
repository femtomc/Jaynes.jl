function learnable_normal(x::Float64, y::Float64)
    l = learnable(:l)
    m = learnable(:m)
    q = rand(:q, Normal(l, 3 + m^2))
    return q
end

@testset "Convergence for learning - MAP 1" begin
    sel = selection([(:q, ) => 6.0])
    params = parameters([(:l, ) => 3.0,
                         (:m, ) => 10.0])
    for i in 1:200
        ret, cl, w = generate(sel, params, learnable_normal, 5.0, 3.0)
        param_grads = get_topeter_gradients(params, cl, 1.0)
        params = update_parameters(ADAM(0.05, (0.9, 0.8)), params, param_grads)
    end
    @test params[:l] ≈ 6.0 atol = 1e-2
    @test params[:m] ≈ 0.0 atol = 1e-2

    # Train.
    params = train(sel, params, learnable_normal, 5.0, 3.0; opt = ADAM(0.05, (0.9, 0.8)), iters = 2000)
    @test params[:l] ≈ 6.0 atol = 1e-2
    @test params[:m] ≈ 0.0 atol = 1e-2
end

function model()
    slope = rand(:slope, Normal(-1, exp(0.5)))
    intercept = rand(:intercept, Normal(1, exp(2.0)))
end

function var()
    slope_mu = learnable(:slope_mu)
    slope_log_std = learnable(:slope_log_std)
    intercept_mu = learnable(:intercept_mu)
    intercept_log_std = learnable(:intercept_log_std)
    slope = rand(:slope, Normal(slope_mu, exp(slope_log_std)))
    intercept = rand(:intercept, Normal(intercept_mu, exp(intercept_log_std)))
end

@testset "Convergence for learning - VI 1" begin
    sel = selection()
    initial_params = parameters([(:slope_mu, ) => 0.0,
                                 (:slope_log_std, ) => 0.0,
                                 (:intercept_mu, ) => 0.0,
                                 (:intercept_log_std, ) => 0.0])
    params, _, _ = advi(sel, initial_params, var, (), model, (); iters = 2000, opt = ADAM(0.005, (0.9, 0.999)))
    @test params[:slope_mu] ≈ -1.0 atol = 7e-2
    @test params[:slope_log_std] ≈ 0.5 atol = 7e-2
    @test params[:intercept_mu] ≈ 1.0 atol = 7e-2
    @test params[:intercept_log_std] ≈ 2.0 atol = 7e-2
end
