This page keeps a set of common model examples expressed in Jaynes.

## Bayesian linear regression

```julia
module BayesianLinearRegression

using Jaynes
using Distributions

function bayeslinreg(N::Int)
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    for x in 1:N
        y = rand(:y => x, Normal(β*x, σ))
    end
end

# Observations
obs = [(:y => 1, 0.9), (:y => 2, 1.7), (:y => 3, 3.2), (:y => 4, 4.3)]
sel = selection(obs)

# SIR
@time ps = importance_sampling(bayeslinreg, (length(obs), ); observations = sel, num_samples = 20000)
num_res = 5000
resample!(ps, num_res)

# Some parameter statistics
mean_σ = sum(map(ps.calls) do cl
    cl[:σ]
end) / num_res
println("Mean σ: $mean_σ")

mean_β = sum(map(ps.calls) do cl
    cl[:β]
end) / num_res
println("Mean β: $mean_β")

end # module
```

## Backpropagation for choices and learnable parameters

```julia
module Learnable

using Jaynes
using Distributions

function foo(q::Float64)
    p = learnable(:l, 10.0)
    z = rand(:z, Normal(p, q))
    return z
end

function bar(x::Float64, y::Float64)
    l = learnable(:l, 5.0)
    m = learnable(:m, 10.0)
    q = rand(:q, Normal(l, y + m))
    f = rand(:f, foo, 5.0)
    return f
end

cl = trace(bar, 5.0, 1.0)

params = get_parameters(cl)
println("Parameters:\n$(params)")

grads = get_parameter_gradients(cl, 1.0)
println("\nParameter gradients:\n$(grads)")

grads = get_choice_gradients(cl, 1.0)
println("\nChoice gradients:\n$(grads)")

end # module
```
