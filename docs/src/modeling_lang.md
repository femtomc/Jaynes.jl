# Modeling language

The modeling language for Jaynes is...Julia! We don't require the use of macros to specify probabilistic models, because the tracer tracks code using introspection at the level of lowered (and IR) code.

However, this doesn't give you free reign to write anything with `rand` calls and expect it to compile to a valid probabilistic program. Here we outline a number of restrictions (which are echoed in [Gen](https://www.gen.dev/dev/ref/gfi/#Mathematical-concepts-1)) which are required to allow inference to stay strictly Bayesian.

1. For branches with address spaces which intersect, the addresses in the intersection _must_ have distributions with the same base measure. This means you cannot swap continuous for discrete or vice versa depending on which branch you're on.

2. Mutable state does not interact well with iterative inference (e.g. MCMC). Additionall, be careful about the support of your distributions in this regard.
