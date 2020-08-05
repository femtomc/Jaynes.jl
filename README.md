<p align="center">
<img height="250px" src="docs/assets/jaynes.png"/>
</p>
<br>

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://femtomc.github.io/Jaynes.jl/dev)
[![Build Status](https://travis-ci.org/femtomc/Jaynes.jl.svg?branch=master)](https://travis-ci.org/femtomc/Jaynes.jl)
[![codecov](https://codecov.io/gh/femtomc/Jaynes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/femtomc/Jaynes.jl)

> Give your program some backbone!

---

```julia
] add Jaynes
```

`Jaynes.jl` is a (research-oriented) universal probabilistic programming framework which uses source-to-source IR transformations and contextual dispatch to implement the core routines for modeling and inference. Succinctly, the core implementation of Jaynes provides the [_generative function interface_](https://www.gen.dev/stable/ref/gfi/#Generative-function-interface-1) of [Gen.jl](https://www.gen.dev/) as a set of configurable [dynamo](https://fluxml.ai/IRTools.jl/latest/dynamo/) execution contexts. These contexts can be used to overload your code with new probabilistic programming/differentiable programming behaviors.

This allows the usage of pure Julia as the primary modeling language:

```julia
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
                    (:y => i, ) => 3.0 * x[i] + randn()
                end)

n_samples = 5000
@time ps, lnw = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))
zipped = zip(ps.calls, lnw)

est_σ = sum(map(zipped) do (cl, w)
                 get_ret(cl[:σ]) * exp(w)
             end)
println("Estimated σ: $est_σ")

est_β = sum(map(zipped) do (cl, w)
                 get_ret(cl[:β]) * exp(w)
             end)
println("Estimated β: $est_β")
```

See [Examples](https://femtomc.github.io/Jaynes.jl/dev/examples/) for some more code snippets - including arbitrary control flow!

## Inference

Jaynes currently supports the following inference algorithms:

1. Importance sampling (with and without custom proposals)
2. Particle filtering (with and without custom proposals)
3. Metropolis-Hastings (with and without custom proposals)
4. ADVI (with [Flux.jl](https://github.com/FluxML/Flux.jl) optimisers, currently uses [Zygote.jl](https://github.com/FluxML/Zygote.jl) for reverse-mode AD)

[Jaynes also supports the integration of differentiable programming with probabilistic programming.](https://femtomc.github.io/Jaynes.jl/dev/library_api/diff_prog/)

These algorithms automatically support adaptive multi-threading, which is enabled by setting your `JULIA_NUM_THREADS` environment variable. See [Multi-Threading](https://docs.julialang.org/en/v1/base/multi-threading/) for more details.

## Foreign model interfaces

[Jaynes supports interoperability with models and inference algorithms expressed in Gen.jl and Soss.jl](https://femtomc.github.io/Jaynes.jl/dev/library_api/fmi/)

In particular, after loading the model interface (using `@load_gen_fmi` or `@load_soss_fmi`), you can write models in `Gen.jl` or `Soss.jl` and have the dynamo tracers construct and reason about specialized call site representations for the interface.

```julia
Jaynes.@load_gen_fmi()

@gen (static) function foo(z::Float64)
    x = @trace(normal(z, 1.0), :x)
    y = @trace(normal(x, 1.0), :y)
    return x
end

Gen.load_generated_functions()

bar = () -> begin
    x = rand(:x, Normal(0.0, 1.0))
    return foreign(:foo, foo, x)
end

ret, cl = Jaynes.simulate(bar)
display(cl.trace)
```

All of Jaynes native inference algorithms will work on the wrapped representation. You can even combine interfaces (with some limitations on inference currently) for great fun and profit.

```julia
Jaynes.@load_soss_fmi()
Jaynes.@load_gen_fmi()

# A Soss model.
m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(5)
end

@gen (static) function foo(x::Float64)
    y = @trace(normal(x, 1.0), :y)
    return y
end
Gen.load_generated_functions()

# A funky model :)
bar = () -> begin
    x = rand(:x, Normal(5.0, 1.0))
    gen_ret = foreign(:gen, foo, x)
    soss_ret = foreign(:foo, m, (σ = x,))
    return soss_ret
end

ret, cl = Jaynes.simulate(bar)
```

## Hack your program!

Jaynes also includes a set of power tools for giving your programs differentiable hooks, and re-configurable parameter sites. `learnable` sites are configurable by `Parameters` instances which can be passed in to execution contexts and trained with backpropagation. Here, Jaynes will learn a parameter value for `:q` with a loss function equivalent to the ["soft evidence"](https://agentmodels.org/chapters/3-agents-as-programs.html) expressed by the special [factor call site](https://femtomc.github.io/Jaynes.jl/dev/library_api/sites/).

```julia
foo = y -> begin
    q = learnable(:q)
    x = q + y
    factor(-x ^ 2)
    y
end

params = parameters([(:q, ) => 1.0])
ps = train(params, foo, 5.0)
display(ps)
```

`fillable` sites are similar to `learnable` sites - but they can't be trained, and won't interfere with gradient-based training of learnable sites. [These sites are accessed using the selection interface.](https://femtomc.github.io/Jaynes.jl/dev/library_api/selection_interface/)

```julia
foo = y -> begin
    q = fillable(:q)
    z = learnable(:z)
    x = (q + z) * y
    factor(-x ^ 2)
end

params = parameters([(:z, ) => 1.0])
sel = selection([(:q, ) => 1.0])
ps = train(sel, params, foo, 5.0)
display(ps)
```
