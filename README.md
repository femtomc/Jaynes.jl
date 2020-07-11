<p align="center">
<img height="250px" src="docs/assets/jaynes.png"/>
</p>
<br>

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://femtomc.github.io/Jaynes.jl/dev)
[![Build Status](https://travis-ci.org/femtomc/Jaynes.jl.svg?branch=master)](https://travis-ci.org/femtomc/Jaynes.jl)
[![codecov](https://codecov.io/gh/femtomc/Jaynes.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/femtomc/Jaynes.jl)

_Jaynes_ is a universal probabilistic programming framework which uses IR transformations and contextual dispatch to implement the core routines for modeling and inference.

Jaynes currently supports the following inference algorithms:

1. Importance sampling (with and without custom proposals)
2. Particle filtering (with and without custom proposals)
3. Metropolis-hastings (with and without custom proposals)

[Jaynes also supports the integration of differentiable programming with probabilistic programming.](https://femtomc.github.io/Jaynes.jl/dev/diff_prog/)
