# Model.
function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    for i in 1 : length(x)
        rand(:y => i, Normal(β * x[i], σ))
    end
end

# Data.
data_len = 100
x = [Float64(i) for i in 1 : data_len]
y = map(x) do k
    3.0 * k + randn()
end
obs = target(map(1 : data_len) do i
                 (:y => i, ) => y[i]
             end)

@testset "Inference library smoke test 1 - Bayesian linear regression" begin

    # Importance sampling.
    is_test = () -> begin
        n_samples = 50000
        ps, nw = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))
        zipped = zip(ps.calls, nw)

        est_σ = sum(map(zipped) do (cl, w)
                        (cl[:σ]) * w
                    end)

        est_β = sum(map(zipped) do (cl, w)
                        (cl[:β]) * w
                    end)
        @test isapprox(0.0, est_β - 3.0, atol=1e-1)
    end

    # MH random walk kernel.
    mh_test = () -> begin
        n_iters = 50000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(target([(:σ, ), (:β, )]), cl)
            i % 50 == 0 && begin
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        @test isapprox(0.0, est_β - 3.0, atol=2e-1)
    end

    # MH random walk kernel with proposal.
    proposal = (cl, y) -> begin
        σ = rand(:σ, Normal(6.0, 1.0))
        β = rand(:β, Normal(mean(y) / length(y), 3.0))
    end

    mh_test_with_proposal = () -> begin
        n_iters = 10000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(cl, proposal, (y, ))
            i % 30 == 0 && begin
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        @test isapprox(0.0, est_β - 3.0, atol=1e-1)
    end

    # ADVI.
    surrogate = () -> begin
        μ₁ = learnable(:μ₁)
        μ₂ = learnable(:μ₂)
        σ₂ = learnable(:σ₂)
        σ = rand(:σ, Normal(1.0 + μ₁, 0.3))
        β = rand(:β, Normal(μ₂, σ₂ ^ 2))
    end

    advi_test = () -> begin
        ps = learnables([(:μ₁, ) => 5.0,
                         (:μ₂, ) => 1.0,
                         (:σ₂, ) => 1.0])
        @time ps, elbows, _ = advi(obs, ps,
                                   surrogate, (),
                                   bayesian_linear_regression, (x, );
                                   opt = ADAM(0.05, (0.9, 0.8)),
                                   iters = 1000)
    end

    # ADGV.
    surrogate = () -> begin
        μ₁ = learnable(:μ₁)
        μ₂ = learnable(:μ₂)
        σ₂ = learnable(:σ₂)
        σ = rand(:σ, Normal(1.0 + μ₁, 0.3))
        β = rand(:β, Normal(μ₂, σ₂ ^ 2))
    end

    adgv_test = () -> begin
        ps = learnables([(:μ₁, ) => 5.0,
                         (:μ₂, ) => 1.0,
                         (:σ₂, ) => 1.0])
        ps, elbows, _ = adgv(obs, ps, 10,
                             surrogate, (),
                             bayesian_linear_regression, (x, );
                             opt = ADAM(0.05, (0.9, 0.8)),
                             iters = 1000)
    end

    # HMC kernel.
    hmc_test = () -> begin
        n_iters = 1000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = hmc(target([(:σ, ), (:β, )]), cl)
            i % 10 == 0 && begin
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        @test isapprox(0.0, est_β - 3.0, atol=2e-1)
    end

    # Combination kernel.
    combo_kernel_test = () -> begin
        n_iters = 1000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(cl, proposal, (y, ))
            cl, _ = hmc(target([(:σ, ), (:β, )]), cl)
            i % 10 == 0 && begin
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        @test isapprox(0.0, est_β - 3.0, atol=1e-1)
    end

    # Boomerang kernel.
    boomerang_test = () -> begin
        n_iters = 500
        ret, cl = generate(obs, bayesian_linear_regression, x)
        sel = get_target(cl)
        sel_values, _ = get_choice_gradients(sel, cl, 1.0)
        d = length(array(sel_values, Float64))
        flow = Boomerang(sparse(I, d, d), zeros(d), 1.0)
        θ = rand(MvNormal(d, 1.0))
        calls = []
        for i in 1 : n_iters
            cl, _ = pdmk(target([(:σ, ), (:β, )]), cl, flow, θ)
            i % 10 == 0 && begin
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end)

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end)
    end

    is_test()
    mh_test()
    mh_test_with_proposal()
    hmc_test()
    combo_kernel_test()
    #@time boomerang_test()
    #@time advi_test() 
    #@time adgv_test() 

end
