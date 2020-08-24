@def title = "Bayesian linear regression"

Here's a simple example of Bayesian linear regression.

```julia:/code/bayeslinreg
using Jaynes

# Model.
function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(1.0, 3.0))
    y = [rand(:y => i, Normal(β * x[i], σ)) for i in 1 : length(x)]
    y
end
```

Because we are constructing a distribution over address maps, we must specify our observations in a form which this distribution understands.

```julia:/code/bayeslinreg
# Data.
data_len = 100
x = [Float64(i) for i in 1 : data_len]
obs = target([(:y => i, ) => 3.0 * x[i] + randn() for i in 1 : data_len])
```

The call to `target` constructs a `DynamicMap{Value}` - a structure which specifies that the specified addresses should be fixed for inference contexts. Here, we're constraining the data observation `y`.

Now we can perform _importance sampling_ - here, we use the prior as the proposal.

```julia:/code/bayeslinreg
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
```

\show{/code/bayeslinreg}
