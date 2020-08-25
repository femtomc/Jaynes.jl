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
