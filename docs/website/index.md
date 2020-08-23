@def title = "Jaynes"
@def tags = ["syntax", "code"]

# Core concepts

**Jaynes.jl** is a probabilistic programming package which emphasizes _ease-of-use_, _interoperability_, and _scalability_.

* _Ease-of-use_ means that new users should be able to quickly understand how to create probabilistic programs, as well as understand how to utilize the inference library to accomplish their tasks. This also means that Jaynes provides extension interfaces to highly optimized inference packages (like AdvancedHMC) for models out of the box.

* _Interoperability_ means that Jaynes plays nicely across the probabilistic programming ecosystem. If you are familiar with Gen, Turing, Soss, even Pyro - Jaynes provides a set of interfaces (the _foreign model interface_) for utilizing models in these other systems. Other systems can be integrated rapidly - by writing an extension module.

* _Scalability_ means that Jaynes can be optimized as you grow into the library. This feature manifests itself as specialized call site representations, and a suite of compiler tools for use in constructing specialized inference algorithms.
