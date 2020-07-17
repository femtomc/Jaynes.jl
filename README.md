<p align="center">
<img height="250px" src="docs/assets/jaynes.png"/>
</p>
<br>

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://femtomc.github.io/Jaynes.jl/dev)
[![Build Status](https://travis-ci.org/femtomc/Jaynes.jl.svg?branch=master)](https://travis-ci.org/femtomc/Jaynes.jl)
[![codecov](https://codecov.io/gh/femtomc/Jaynes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/femtomc/Jaynes.jl)

> This is alpha software!

_Jaynes_ is a (research-oriented) universal probabilistic programming framework which uses IR transformations and contextual dispatch to implement the core routines for modeling and inference. This allows the usage of pure Julia as the primary modeling language:

```julia
function kernel(prev_latent::Float64)
    z = rand(:z, Normal(prev_latent, 1.0))
    if z > 2.0
        x = rand(:x, Normal(z, 3.0))
    else
        x = rand(:x, Normal(z, 10.0))
    end
    return z
end

# Specialized Markov call informs tracer of dependency information to accelerate inference.
kernelize = n -> markov(:k, kernel, n, 5.0)
```

Jaynes currently supports the following inference algorithms:

1. Importance sampling (with and without custom proposals)
2. Particle filtering (with and without custom proposals)
3. Metropolis-Hastings (with and without custom proposals)
4. ADVI (with Flux optimisers, currently uses Zygote for reverse-mode AD)

[Jaynes also supports the integration of differentiable programming with probabilistic programming.](https://femtomc.github.io/Jaynes.jl/dev/library_api/diff_prog/)
