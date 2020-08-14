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
        println("Importance sampling:")
        n_samples = 5000
        ps, lnw = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))
        zipped = zip(ps.calls, lnw)

        est_σ = sum(map(zipped) do (cl, w)
                        (cl[:σ]) * exp(w)
                    end)
        println("Estimated σ: $est_σ")

        est_β = sum(map(zipped) do (cl, w)
                        (cl[:β]) * exp(w)
                    end)
        println("Estimated β: $est_β")
    end

    # MH random walk kernel.
    mh_test = () -> begin
        println("\nRandom walk Metropolis-Hastings:")
        n_iters = 5000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(target([(:σ, ), (:β, )]), cl)
            i % 30 == 0 && begin
                println("σ => $((cl[:σ])), β => $((cl[:β]))")
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)
        println("Estimated σ: $est_σ")

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        println("Estimated β: $est_β")
    end

    # MH random walk kernel with proposal.
    proposal = (cl, y) -> begin
        σ = rand(:σ, Normal(6.0, 1.0))
        β = rand(:β, Normal(mean(y) / length(y), 3.0))
    end

    mh_test_with_proposal = () -> begin
        println("\nMetropolis-Hastings with custom proposal:")
        n_iters = 10000
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(cl, proposal, (y, ))
            i % 30 == 0 && begin
                println("σ => $((cl[:σ])), β => $((cl[:β]))")
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)
        println("Estimated σ: $est_σ")

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        println("Estimated β: $est_β")
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
        println("\nADVI:")
        ps = learnables([(:μ₁, ) => 5.0,
                         (:μ₂, ) => 1.0,
                         (:σ₂, ) => 1.0])
        @time ps, elbows, _ = advi(obs, ps,
                                   surrogate, (),
                                   bayesian_linear_regression, (x, );
                                   opt = ADAM(0.05, (0.9, 0.8)),
                                   iters = 1000)
        display(ps)
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
        println("\nADGV:")
        ps = learnables([(:μ₁, ) => 5.0,
                         (:μ₂, ) => 1.0,
                         (:σ₂, ) => 1.0])
        ps, elbows, _ = adgv(obs, ps, 10,
                             surrogate, (),
                             bayesian_linear_regression, (x, );
                             opt = ADAM(0.05, (0.9, 0.8)),
                             iters = 1000)
        display(ps)
    end

    # HMC kernel.
    hmc_test = () -> begin
        println("\nHamiltonian Monte Carlo:")
        n_iters = 500
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = hmc(target([(:σ, ), (:β, )]), cl)
            i % 10 == 0 && begin
                println("σ => $((cl[:σ])), β => $((cl[:β]))")
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)
        println("Estimated σ: $est_σ")

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        println("Estimated β: $est_β")
    end

    # Combination kernel.
    combo_kernel_test = () -> begin
        println("\nCombo kernel special:")
        n_iters = 500
        ret, cl = generate(obs, bayesian_linear_regression, x)
        calls = []
        for i in 1 : n_iters
            cl, _ = mh(cl, proposal, (y, ))
            cl, _ = hmc(target([(:σ, ), (:β, )]), cl)
            i % 10 == 0 && begin
                println("σ => $((cl[:σ])), β => $((cl[:β]))")
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end) / length(calls)
        println("Estimated σ: $est_σ")

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end) / length(calls)
        println("Estimated β: $est_β")
    end

    # Boomerang kernel.
    boomerang_test = () -> begin
        println("\nBoomerang sampler:")
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
                println("σ => $((cl[:σ])), β => $((cl[:β]))")
                push!(calls, cl)
            end
        end

        est_σ = sum(map(calls) do cl
                        (cl[:σ])
                    end)
        println("Estimated σ: $est_σ")

        est_β = sum(map(calls) do cl
                        (cl[:β])
                    end)
        println("Estimated β: $est_β")
    end

    @time is_test()
    @time mh_test()
    @time mh_test_with_proposal()
    @time hmc_test()
    @time combo_kernel_test()
    #@time boomerang_test()
    #@time advi_test() 
    #@time adgv_test() 

end
