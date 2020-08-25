module BayesianLinearRegression

using Jaynes

# Model.
function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(1.0, 3.0))
    y = [rand(:y => i, Normal(β * x[i], σ)) for i in 1 : length(x)]
    y
end

# Data.
data_len = 100
x = [Float64(i) for i in 1 : data_len]
obs = target([(:y => i, ) => 3.0 * x[i] + randn() for i in 1 : data_len])

# Importance sampling.
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

end # module
