module BayesianLinearRegression

include("../src/Jaynes.jl")
using .Jaynes

function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    for i in 1 : length(x)
        rand(:y => i, Normal(β * x[i], σ))
    end
end

data_len = 50
x = [Float64(i) for i in 1 : data_len]
obs = selection(map(1 : data_len) do i
                    (:y => i, ) => 3.0 * x[i] + randn()
                end)

# Importance sampling.
is_test = () -> begin
    println("Importance sampling:")
    n_samples = 5000
    ps, lnw = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))
    zipped = zip(ps.calls, lnw)

    est_σ = sum(map(zipped) do (cl, w)
                    get_ret(cl[:σ]) * exp(w)
                end)
    println("Estimated σ: $est_σ")

    est_β = sum(map(zipped) do (cl, w)
                    get_ret(cl[:β]) * exp(w)
                end)
    println("Estimated β: $est_β")
end
@time is_test()

# HMC kernel.
hmc_test = () -> begin
    println("\nHamiltonian Monte Carlo:")
    n_iters = 1000
    ret, cl = generate(obs, bayesian_linear_regression, x)
    calls = []
    for i in 1 : n_iters
        cl, _ = hmc(selection([(:σ, ), (:β, )]), cl)
        i % 30 == 0 && begin
            println("σ => $(get_ret(cl[:σ])), β => $(get_ret(cl[:β]))")
            push!(calls, cl)
        end
    end

    est_σ = sum(map(calls) do cl
                    get_ret(cl[:σ])
                end) / length(calls)
    println("Estimated σ: $est_σ")

    est_β = sum(map(calls) do cl
                    get_ret(cl[:β])
                end) / length(calls)
    println("Estimated β: $est_β")
end
@time hmc_test()

# Boomerang kernel.
boomerang_test = () -> begin
    println("\nBoomerang sampler:")
    n_iters = 5000
    ret, cl = generate(obs, bayesian_linear_regression, x)
    calls = []
    for i in 1 : n_iters
        @time cl, _ = boo(selection([(:σ, ), (:β, )]), cl)
        i % 100 == 0 && push!(call, cl)
    end

    est_σ = sum(map(calls) do cl
                    get_ret(cl[:σ])
                end)
    println("Estimated σ: $est_σ")

    est_β = sum(map(calls) do cl
                    get_ret(cl[:β])
                end)
    println("Estimated β: $est_β")
end
boomerang_test()

end # module
