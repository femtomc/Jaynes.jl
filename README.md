<p align="center">
<img height="250px" src="img/walkman.jpeg"/>
</p>
<br>

_Walkman_ is a probabilistic programming framework which uses IR transformations to implement the core routines for modeling and inference.

Currently, _Walkman_ supports a dynamic modeling DSL which is syntactically close (and semantically equivalent) to the dynamic DSL in [Gen](https://www.gen.dev/). It is _unoptimized_. There are a few _effects_ (which are semantically similar to Gen's combinators) but their implementation via contextual dispatch is not complete nor stable. This will likely improve with time, as I settle on representations which allow the sort of analysis I'm aiming for.

Currently supported inference algorithms for this DSL:
- [X] Importance sampling
- [ ] Programmable MCMC
- [ ] HMC
- [ ] Metropolis-Hastings
- [ ] Particle filtering

_Walkman_ also aims to support a restricted _graph-based_ DSL which allows the user to utilize graphical model inference algorithms. This is a WIP, and requires a bit more research at the IR level.

## Other notes

The motivation for this project is to identify interfaces and techniques to combine programmable inference with graphical model inference. These techniques have complementary strengths and weaknesses - programmable sampling algorithms tend to have difficulties in high-dimensions (but can answer joint queries about a model efficiently when they are efficient) whereas the asymptotic complexity of graphical model algorithms is typically not dependent on the dimensionality of the model (and instead depends on the topology of the dependence graph) but queries are typically restricted to be marginal queries.

## Examples

```julia
module Geometric

include("../src/Walkman.jl")
using .Walkman
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)

tr, score = trace(geo, (0.3, ))
display(tr)

end # module

#  __________________________________
#
#               Playback
#
# :geo => :flip
#          val  = 0
#
# flip
#          val  = 0
#
# :geo => (:geo => :flip)
#          val  = 1
#
#  __________________________________
#
# score : 0.0
#
# func : typeof(Main.Geometric.geo)
#
# args : Tuple{Float64}
#
# retval : 2
#
#  __________________________________
```

## Acknowledgements

The ideas which are going into this package would not have been possible without numerous conversations with very smart people in the Julia community. I would like to acknowledge the following people

> Jarred Barber, Alex Lew, Marco Cusumano-Towner, Jarrett Revels, Valentin Churavy, George Matheos, Chad Scherrer, Martin Trapp, Philipp Gabler, Lyndon White, Mike Innes, and Ari Katz...amongst many others in the probabilistic programming community.

as well as the following systems

> [Gen](https://github.com/probcomp/Gen.jl), [Turing](https://github.com/TuringLang/Turing.jl), [Soss](https://github.com/cscherrer/Soss.jl), [Pyro](https://pyro.ai/), [Unison](https://www.unisonweb.org/), [Cassette](https://github.com/jrevels/Cassette.jl) and [Zygote](https://github.com/FluxML/Zygote.jl).


---
