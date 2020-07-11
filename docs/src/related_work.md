Jaynes is a _context-oriented programming_ system for probabilistic programming. Internally, the current implementation closely follows the design of [Gen](https://www.gen.dev/) which also uses the notion of stateful execution contexts to produce the interfaces required for inference. 

In contrast to Gen (which provides powerful optimizations for programs written in the [static DSL](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1)), Jaynes is focused on an optimized dynamic language which allows most of the Julia language to be used in expressing probabilistic programs.

Jaynes uses many concepts from the design and implementation of Gen. First and foremost, I would recommend users of Jaynes become familiar with Gen - to understand the problems which Jaynes attempts to solve. The following papers may be useful in this regard:

1. [Gen: a general-purpose probabilistic programming system with programmable inference](https://dl.acm.org/doi/10.1145/3314221.3314642)
2. [Probabilistic programming with programmable inference](https://people.csail.mit.edu/rinard/paper/pldi18.pdf)
3. [A new approach to probabilistic programming inference](http://proceedings.mlr.press/v33/wood14.pdf)
4. [Lightweight Implementations of probabilistic programming languages via transformational compilation](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)

In the design space of compiler metaprogramming tools, the following systems have been highly influential in the design of Jaynes

1. [IRTools](https://github.com/FluxML/IRTools.jl)
2. [Cassette](https://github.com/jrevels/Cassette.jl)

In particular, `IRTools` provides thecore infrastructure for the implementation. Strictly speaking, `Jaynes` is not dependent on some fundamental mechanism which `IRTools` provides (only _generated functions_ from Julia) but `IRTools` greatly reduces the level of risk in working with generated functions and lowered code.

Jaynes has also been influenced by [Turing](https://turing.ml/dev/), the [Poutine effects system](https://docs.pyro.ai/en/stable/poutine.html) in Pyro, and [Unison lang](https://www.unisonweb.org/). Jaynes does not implement _algebraic effects_ in a rigorous (or static!) way, but the usage of execution contexts which control how certain method calls are executed is closely aligned with these concepts.

Finally, the probabilistic programming community in Julia is largely responsible for many of the ideas and conversations which lead to Jaynes. I'd like to thank Chad Scherrer, Martin Trapp, Alex Lew, Jarred Barber, George Matheos, Marco Cusumano-Towner, Ari Katz, Philipp Gabler, Valentin Churavy, Mike Innes, and Lyndon White for auxiliary help and discussion concerning the design and implementation of many parts of the system.
