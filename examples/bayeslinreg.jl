module BayesianLinearRegression

include("../src/Jaynes.jl")
using .Jaynes

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

# ADVI.
surrogate = () -> begin
    μ₁ = learnable(:μ₁)
    μ₂ = learnable(:μ₂)
    σ₂ = learnable(:σ₂)
    σ = rand(:σ, Normal(μ₁, 0.3))
    β = rand(:β, Normal(μ₂, σ₂))
end

advi_test = () -> begin
    println("\nADVI:")
    ps = learnables([(:μ₁, ) => 5.0,
                     (:μ₂, ) => 1.0,
                     (:σ₂, ) => 1.0])
    @time ps, elbows, _ = advi(obs, 
                               ps,
                               surrogate, (),
                               bayesian_linear_regression,
                               (x, );
                               opt = ADAM(0.01, (0.9, 0.8)),
                               iters = 2000)
    println(elbows)
    display(ps)
end
@time advi_test() 
# HMC kernel.
hmc_test = () -> begin
    println("\nHamiltonian Monte Carlo:")
    n_iters = 500
    ret, cl = generate(obs, bayesian_linear_regression, x)
    calls = []
    for i in 1 : n_iters
        cl, _ = hmc(selection([(:σ, ), (:β, )]), cl)
        i % 10 == 0 && begin
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
    n_iters = 500
    ret, cl = generate(obs, bayesian_linear_regression, x)
    calls = []
    for i in 1 : n_iters
        cl, _ = boo(selection([(:σ, ), (:β, )]), cl)
        i % 10 == 0 && begin
            println("σ => $(get_ret(cl[:σ])), β => $(get_ret(cl[:β]))")
            push!(calls, cl)
        end
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
#boomerang_test()

end # module
