function learnable_normal(x::Float64, y::Float64)
    l = learnable(:l)
    m = learnable(:m)
    q = rand(:q, Normal(l, 3 + m^2))
    return q
end

@testset "Convergence for learning - MLE 1" begin
    sel = target([(:q, ) => 6.0])
    params = learnables([(:l, ) => 3.0,
                         (:m, ) => 10.0])
    ret, cl, w = generate(sel, params, learnable_normal, 5.0, 3.0)
    for i in 1 : 40
        δ, params, acc = mle(params, cl)
        δ < 1e-12 && acc && break
    end
    @test params[:l] ≈ 6.0 atol = 1e-2
    @test params[:m] ≈ 0.0 atol = 1e-2
end

@testset "Convergence for learning - MAP 1" begin
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

@testset "Convergence for learning - ADVI 1" begin
    sel = target()
    params = learnables([(:slope_mu, ) => 0.0,
                         (:slope_log_std, ) => 0.0,
                         (:intercept_mu, ) => 0.0,
                         (:intercept_log_std, ) => 0.0])
    opt = ADAM(0.005, (0.9, 0.999))
    for i in 1 : 500
        params, _ = advi(opt, sel, params, var, (), model, ())
    end
    @test params[:slope_mu] ≈ -1.0 atol = 7e-2
    @test params[:slope_log_std] ≈ 0.5 atol = 7e-2
    @test params[:intercept_mu] ≈ 1.0 atol = 7e-2
    @test params[:intercept_log_std] ≈ 2.0 atol = 7e-2
end

#@testset "Convergence for learning - geo VIMCO 1" begin
#    sel = target()
#    params = learnables([(:slope_mu, ) => 0.0,
#                         (:slope_log_std, ) => 0.0,
#                         (:intercept_mu, ) => 0.0,
#                         (:intercept_log_std, ) => 0.0])
#    opt = ADAM(0.005, (0.9, 0.999))
#    for i in 1 : 500
#        params, _ = adgv(opt, sel, params, var, (), model, ())
#    end
#    @test params[:slope_mu] ≈ -1.0 atol = 7e-2
#    @test params[:slope_log_std] ≈ 0.5 atol = 7e-2
#    @test params[:intercept_mu] ≈ 1.0 atol = 7e-2
#    @test params[:intercept_log_std] ≈ 2.0 atol = 7e-2
#end
