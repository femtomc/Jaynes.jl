@def title = "E.T. Jaynes home phone"
@def tags = ["probabilistic programming", "programmable inference"]

**Jaynes.jl** (Jaynes) is a probabilistic programming framework based on a compiler interception version of [the generative function interface of Gen.jl](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1)[^1].

> This package is in open-alpha. Expect some bumps, especially as [new compiler interfaces](https://github.com/Keno/Compiler3.jl) stabilize in Julia `VERSION` > 1.6.

---

Jaynes emphasizes _ease-of-use_, _interoperability_, and _scalability_.

* _Ease-of-use_ means that new users should be able to quickly understand how to create probabilistic programs, as well as understand how to utilize the inference library to accomplish their tasks. This also means that Jaynes provides extension interfaces to highly optimized inference packages (like [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)) for models out of the box.

* _Interoperability_ means that Jaynes plays nicely across the probabilistic programming ecosystem. If you are familiar with [Gen.jl](https://www.gen.dev/), [Turing.jl](https://turing.ml/dev/), [Soss.jl](https://github.com/cscherrer/Soss.jl), even [Pyro](https://pyro.ai/)[^2] - Jaynes provides a set of interfaces (the _foreign model interface_) for utilizing models in these other systems. Other systems can be integrated rapidly - by writing an extension module.

* _Scalability_ means that models and inference programs expressed using Jaynes can be optimized as you grow into the framework. This goal manifests itself in a few ways: specialized call site representations, a suite of compiler tools for use in constructing specialized inference algorithms, and [programmable inference](https://people.csail.mit.edu/rinard/paper/pldi18.pdf)[^3].


---

[^1]: Roughly, this interface describes the set of capabilities which, when implemented for a model class, allows for the construction of customizable sampling-based inference algorithms. This idea originally appeared under _stochastic procedure interface_ in [Venture](https://arxiv.org/abs/1404.0099).

[^2]: Pyro compatibility is provided by an open-source wrapper [Pyrox.jl](https://github.com/femtomc/Pyrox.jl) which operates through [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).

[^3]: This terminology started with [Venture](http://probcomp.csail.mit.edu/software/venture/). The most mature representation of a system which supports this functionality is [Gen.jl](https://github.com/probcomp/Gen.jl).
