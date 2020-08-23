@def title = "Jaynes"
@def tags = ["syntax", "code"]

**Jaynes.jl** (Jaynes) is a probabilistic programming framework [based on the generative function interface of Gen.jl](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1)
which emphasizes _ease-of-use_, _interoperability_, and _scalability_.

* _Ease-of-use_ means that new users should be able to quickly understand how to create probabilistic programs, as well as understand how to utilize the inference library to accomplish their tasks. This also means that Jaynes provides extension interfaces to highly optimized inference packages (like [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)) for models out of the box.

* _Interoperability_ means that Jaynes plays nicely across the probabilistic programming ecosystem. If you are familiar with [Gen.jl](https://www.gen.dev/), [Turing.jl](https://turing.ml/dev/), [Soss.jl](https://github.com/cscherrer/Soss.jl), even [Pyro](https://pyro.ai/) - Jaynes provides a set of interfaces (the _foreign model interface_) for utilizing models in these other systems. Other systems can be integrated rapidly - by writing an extension module.

* _Scalability_ means that models expressed using Jaynes can be optimized as you grow into the framework. This feature manifests itself as specialized call site representations, and a suite of compiler tools for use in constructing specialized inference algorithms.

> This package is an open-alpha package. Expect some bumps, especially as new compiler interfaces stabilize in Julia `VERSION` > 1.6.
