This page keeps a set of small examples expressed in Jaynes.

## Geometric program

```julia
module Geometric

using Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 1 : 1 + rand(:geo, geo, p)

ret, cl = simulate(geo, 0.2)
display(cl.trace)

end # module
```

## Bayesian linear regression

This example uses the `plate` special language feature (equivalent to plate notation) to perform IID draws from the `Normal` specified in the program. This is more concise than a loop, and also allows inference algorithms to utilize efficient operations for updating and regenerating choices.

```julia
module BayesianLinearRegression

using Jaynes
using Distributions

function bayesian_linear_regression(N::Int)
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = plate(:y, x -> rand(:draw, Normal(β*x, σ)), [Float64(i) for i in 1 : N])
end

# Example trace.
ret, cl = simulate(bayesian_linear_regression, 10)
display(cl.trace)

sel = selection(map(1 : 100) do i
                    (:y => i => :draw, Float64(i) + rand())
                end)

@time ps, ret = importance_sampling(sel, 5000, bayesian_linear_regression, (100, ))

num_res = 1000
resample!(ps, num_res)
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

## Specialized Markov call sites

This example illustrates how specialized call sites (accessed through the tracer language features) can be used to accelerate inference.

```julia
module HiddenMarkovModel

using Jaynes
using Distributions

function kernel(prev_latent::Float64)
    z = rand(:z, Normal(prev_latent, 1.0))
    x = rand(:x, Normal(z, 3.0))
    return z
end

# Specialized Markov call site informs tracer of dependency information.
kernelize = n -> markov(:k, kernel, n, 5.0)

simulation = () -> begin
    ps = initialize_filter(selection(), 5000, kernelize, (1, ))
    for i in 2:50
        sel = selection((:k => i => :x, 5.0))

        # Complexity of filter step is constant as a size of the trace.
        @time filter_step!(sel, ps, NoChange(), (i,))
    end
    return ps
end

ps = simulation()

end # module
```
