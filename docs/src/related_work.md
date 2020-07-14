## Probabilistic programming

There are a number of good references in both textbook and research paper form on probabilistic programming. In reference to this system, here are a few I would recommend:

1. [An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756)
2. [A compilation target for probabilistic programming languages](http://proceedings.mlr.press/v32/paige14.pdf)
4. [Lightweight implementations of probabilistic programming languages via transformational compilation](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)
5. [Probabilistic programming](http://human-centered.ai/wordpress/wp-content/uploads/2016/10/GORDON-HENZINGER-NORI-RAJAMANI-2014-Probabilistic-Programming.pdf)
6. [The design and implementation of probabilistic programming languages](http://dippl.org/)
7. [The principles and practice of probabilistic programming](https://web.stanford.edu/~ngoodman/papers/POPL2013-abstract.pdf)

There are also numerous papers about current systems:

1. [Figaro: an object-oriented probabilistic programming language](https://pdfs.semanticscholar.org/0bec/492d110c0746cb3e4dbdf411007ec0bc8772.pdf)
2. [Turing: a language for flexible probabilistic inference](http://proceedings.mlr.press/v84/ge18b/ge18b.pdf)
3. [Venture: a higher-order probabilistic programming platform with programmable inference](https://arxiv.org/abs/1404.0099)
4. [Probabilistic inference by program transformation in Hakaru](https://link.springer.com/chapter/10.1007%2F978-3-319-29604-3_5)
5. [Gen: a general-purpose probabilistic programming system with programmable inference](https://dl.acm.org/doi/10.1145/3314221.3314642)
6. [FACTORIE: probabilistic programming via imperatively defined factor graphs](https://papers.nips.cc/paper/3654-factorie-probabilistic-programming-via-imperatively-defined-factor-graphs.pdf)
7. (monad-bayes) [Practical probabilistic programming with monads](https://dl.acm.org/doi/10.1145/2887747.2804317)
8. (probabilistic C) [A compilation target for probabilistic programming languages](http://proceedings.mlr.press/v32/paige14.pdf)

## Implementation and design

Jaynes is a _context-oriented programming_ system for probabilistic programming. Internally, the current implementation closely follows the design of the dynamic DSL in [Gen](https://www.gen.dev/) which also uses the notion of stateful execution contexts to produce the interfaces required for inference. Jaynes is focused on an optimized dynamic language which allows most of the Julia language to be used in expressing probabilistic programs.

I would recommend users of Jaynes become familiar with Gen - to understand the problems which Jaynes attempts to solve. The following papers may be useful in this regard:

1. [Gen: a general-purpose probabilistic programming system with programmable inference](https://dl.acm.org/doi/10.1145/3314221.3314642)
2. [Probabilistic programming with programmable inference](https://people.csail.mit.edu/rinard/paper/pldi18.pdf)
3. [A new approach to probabilistic programming inference](http://proceedings.mlr.press/v33/wood14.pdf)
4. [Lightweight implementations of probabilistic programming languages via transformational compilation](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)

In the design space of compiler metaprogramming tools, the following systems have been highly influential in the design of Jaynes

1. [IRTools](https://github.com/FluxML/IRTools.jl)
2. [Cassette](https://github.com/jrevels/Cassette.jl)

In particular, `IRTools` provides a large part of the core infrastructure for the implementation. Strictly speaking, `Jaynes` is not dependent on a fundamental mechanism which only `IRTools` provides (anything can be expressed with _generated functions_ from Julia) but `IRTools` greatly reduces the level of risk in working with generated functions and lowered code.

Jaynes has also been influenced by [Turing](https://turing.ml/dev/), the [Poutine effects system](https://docs.pyro.ai/en/stable/poutine.html) in Pyro, and [Unison lang](https://www.unisonweb.org/). Jaynes does not implement _algebraic effects_ in a rigorous (or static!) way, but the usage of execution contexts which control how certain method calls are executed is closely aligned with these concepts.

Finally, the probabilistic programming community in Julia is largely responsible for many of the ideas and conversations which lead to Jaynes. I'd like to thank Chad Scherrer, Martin Trapp, Alex Lew, Jarred Barber, George Matheos, Marco Cusumano-Towner, Ari Katz, Philipp Gabler, Valentin Churavy, Mike Innes, and Lyndon White for auxiliary help and discussion concerning the design and implementation of many parts of the system.
