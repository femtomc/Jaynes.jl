This page keeps a set of small examples expressed in Jaynes.

## Geometric program

```julia
module Geometric

using Jaynes

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

function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = Vector{Float64}(undef, length(x))
    for i in 1 : length(x)
        push!(y, rand(:y => i, Normal(β * x[i], σ)))
    end
    return y
end

x = [Float64(i) for i in 1 : 100]
obs = selection(map(1 : 100) do i
                    (:y => i, ) => 3.0 * x[i] + rand()
                end)

n_samples = 5000
@time ps, lnw = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))

mean_σ = sum(map(ps.calls) do cl
                 get_ret(cl[:σ])
             end) / n_samples
println("Mean σ: $mean_σ")

mean_β = sum(map(ps.calls) do cl
                 get_ret(cl[:β])
             end) / n_samples
println("Mean β: $mean_β")

end # module
```

## Backpropagation for choices and learnable parameters

```julia
module Learnable

using Jaynes

function foo(q::Float64)
    p = learnable(:l)
    z = rand(:z, Normal(p, q))
    return z
end

function bar(x::Float64, y::Float64)
    l = learnable(:l)
    m = learnable(:m)
    q = rand(:q, Normal(l, y + m))
    f = rand(:f, foo, 5.0)
    return f
end

cl = trace(bar, 5.0, 1.0)

ps = parameters([(:l, ) => 5.0,
                 (:m, ) => 5.0,
                 (:f, :l) => 5.0])

grads = get_parameter_gradients(ps, cl, 1.0)
println("\nParameter gradients:\n$(grads)")

grads = get_choice_gradients(ps, cl, 1.0)
println("\nChoice gradients:\n$(grads)")

end # module
```

## Specialized Markov call sites

This example illustrates how specialized call sites (accessed through the tracer language features) can be used to accelerate inference.

```julia
module HiddenMarkovModel

using Jaynes

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
