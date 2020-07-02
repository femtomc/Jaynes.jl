# Modeling language

The modeling language for Jaynes is...Julia! We don't require the use of macros to specify probabilistic models, because the tracer tracks code using introspection at the level of lowered (and IR) code.

However, this doesn't give you free reign to write anything with `rand` calls and expect it to compile to a valid probabilistic program. Here we outline a number of restrictions (which are echoed in [Gen](https://www.gen.dev/dev/ref/gfi/#Mathematical-concepts-1)) which are required to allow inference to stay strictly Bayesian.

1. For branches with address spaces which intersect, the addresses in the intersection _must_ have distributions with the same base measure. This means you cannot swap continuous for discrete or vice versa depending on which branch you're on.

2. Mutable state does not interact well with iterative inference (e.g. MCMC). Additionally, be careful about the support of your distributions in this regard. If you're going to use mutable state in your programs, use `rand` calls in a lightweight manner - only condition on distributions with constant support and be careful about MCMC.

# Vectorized call sites

Jaynes also offers a set of primitive language features for creating _vectorized call sites_ which are similar to the combinators of [Gen](https://www.gen.dev/dev/ref/gfi/#Mathematical-concepts-1). These special features are treated as simple "functional" higher-order functions

```julia
using Jaynes

function bar(m, n)
    x = rand(:x, Normal(m, 1.0))
    q = rand(:q, Normal(x, 5.0))
    z = rand(:z, Normal(n + q, 3.0))
    return q, z
end

function foo()
    x = foldr(rand, :x, bar, 10, 0.3, 3.0)
    y = map(rand, :y, bar, x)
    return y
end

cl = trace(foo)
display(cl.trace; show_values = true)
```

the `foldr` and `map` calls indicate to the tracer that the generation of randomness conforms to a computation pattern which can be vectorized. This allows the tracer to construct an efficient `VectorizedCallSite` which is easier to update than a "black-box" `CallSite` where the dependency information may not be known. 

This is a simple way for the user to increase the efficiency of inference algorithms, by informing the tracer of information which it can't derive on its own (at least for now 😺).

`foldr` requires that the user provide a function `f` with

```math
f: (X, Y) \rightarrow (X, Y)
```

as well as a first argument which denotes the number of fold operations to compute (in the example above, `10`). `foldr` will then iteratively compute the function, passing the return value as arguments to the next computation (from left to right).

`map` does not place requirements on the function `f` (other than the implicit requirements for valid programs, as described above) but does require that the arguments be a `Vector` with each element matching the signature of `f`. `map` then iteratively applies the function as a kernel for each element in the argument vector.

These functions are actually "higher-order" so you can do wild things like

```julia
y = map(rand, :y, bar, foldr(rand, :x, bar, 10, 0.3, 3.0))
```

and expect it to work.
