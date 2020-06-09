There are many active probabilistic programming frameworks in the Julia ecosystem (see [Related Work](related_work.md)) - the ecosystem is one of the richest sources of probabilistic programming research in any language. Frameworks tend to differentiate themselves based upon what model class they efficiently express ([Stheno](https://github.com/willtebbutt/Stheno.jl) for example allows for convenient expression of Gaussian processes). Other frameworks support universal probabilistic programming with sample-based methods, and have optimized features which allow the efficient composition/expression of inference queries (e.g. [Turing](https://turing.ml/dev/) and [Gen](https://github.com/probcomp/Gen.jl)). Jaynes sits within this latter camp - it is strongly influenced by Turing and Gen, but more closely resembles a system like [Zygote](https://github.com/FluxML/Zygote.jl). The full-scope Jaynes system will allow you to express the same things you might express in these other systems - but the long term research goals may deviate slightly from these other libraries. In this section, I will discuss a few of the long term goals.

## Graphical model DSLs 

One of the research goals of Jaynes to identify _composable interfaces_ for allowing users to express static graphical models alongside dynamic sample-based models. This has previously been a difficult challenge - the representations which each class of probabilistic programming system utilizes is very different. Universal probabilistic programming systems have typically relied on sample-based inference, where the main representation is a structured form of an execution trace. In contrast, graphical model systems reason explicitly about distributions and thus require an explicit graph representation of how random variates depend on one another.

A priori, there is no reason why these representations can't be combined in some way. The difficulty lies in deciding how to switch between representations when a program is amenable to both, as well as how the different representations will communicate across inference interfaces. For example, consider performing belief propagation on a model which supports both discrete distributions and function call sites for probabilistic programs which required a sample-based tracing mechanism for interpretation. To enable inference routines to operate on this "call graph" style representation, we have to construct and reason about the representation separately from the runtime of each program.

## Density compilation

TODO.

## Automatic inference compilation

Jaynes already provides (rudimentary) support for gradient-based learning in probabilistic programs. Jaynes also provides a simple interface to construct and use _inference compilers_.

## Black-box extensions

---

To facilitate these research goals, Jaynes is designed as a type of compiler plugin. In contrast to existing frameworks, Jaynes does not require the use of specialized macros to denote where the modeling language begins and ends. The use of macros to denote a language barrier has a number of positive advantages from a user-facing perspective, but some disadvantages related to composability. As an opinion, I believe that a general framework for expressing probabilistic programs should mimic the philosophy of _differentiable programming_. The compiler plugin backend should prevent users from writing programs which are "not valid" (either as a static analysis or a runtime error) but should otherwise get out of the way of the user. Any macros present in the Jaynes library extend the core functionality or provide convenient access to code generation for use by a user - but are not required for modeling and inference.

Because Jaynes is a compiler plugin, it is highly configurable. The goal of the core package is to implement a set of "sensible defaults" for common use, while allowing the implementation of other DSLs, custom inference algorithms, custom representations, etc on top. In this philosophy, Jaynes follows a path first laid out by Gen and Zygote...with a few twists.

Bon appétit!
