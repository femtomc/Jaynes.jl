<p align="center">
<img height="250px" src="docs/assets/jaynes.png"/>
</p>
<br>

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://femtomc.github.io/Jaynes.jl/dev)
[![Build Status](https://travis-ci.org/femtomc/Jaynes.jl.svg?branch=master)](https://travis-ci.org/femtomc/Jaynes.jl)
[![codecov](https://codecov.io/gh/femtomc/Jaynes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/femtomc/Jaynes.jl)

> This is alpha software!

---

```julia
] add Jaynes
```

_Jaynes_ is a (research-oriented) universal probabilistic programming framework which uses IR transformations and contextual dispatch to implement the core routines for modeling and inference. This allows the usage of pure Julia as the primary modeling language:

```julia
function bayesian_linear_regression(N::Int)
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = plate(:y, x -> rand(:draw, Normal(β*x, σ)), [Float64(i) for i in 1 : N])
end

obs = selection(map(1 : 100) do i
                    (:y => i => :draw, Float64(i) + rand())
                end)

ps, ret = importance_sampling(obs, 5000, bayesian_linear_regression, (100, ))
```

See [Examples](https://femtomc.github.io/Jaynes.jl/dev/examples/) for some more code snippets - including arbitrary control flow!

---

Jaynes currently supports the following inference algorithms:

1. Importance sampling (with and without custom proposals)
2. Particle filtering (with and without custom proposals)
3. Metropolis-Hastings (with and without custom proposals)
4. ADVI (with [Flux.jl](https://github.com/FluxML/Flux.jl) optimisers, currently uses [Zygote.jl](https://github.com/FluxML/Zygote.jl) for reverse-mode AD)

[Jaynes also supports the integration of differentiable programming with probabilistic programming.](https://femtomc.github.io/Jaynes.jl/dev/library_api/diff_prog/)

Beginning in version 0.1.4, these algorithms automatically support adaptive multi-threading - enabled by setting your `JULIA_NUM_THREADS` environment variable. See [Multi-Threading](https://docs.julialang.org/en/v1/base/multi-threading/) for more details.
