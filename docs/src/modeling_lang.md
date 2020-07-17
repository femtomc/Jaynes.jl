The modeling language for Jaynes is ... Julia! We don't require the use of macros to specify probabilistic models, because the tracer tracks code using introspection at the level of lowered (and IR) code.

```julia
function bayeslinreg(N::Int)
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = Vector{Float64}(undef, N)
    for i in 1:N
        y[i] = rand(:y => x, Normal(β*x, σ))
    end
    return y
end
```

However, this doesn't give you free reign to write anything with `rand` calls and expect it to compile to a valid probabilistic program. Here we outline a number of restrictions (originally formalized in [Gen](https://www.gen.dev/dev/ref/gfi/#Mathematical-concepts-1)) which are required to allow inference to stay strictly Bayesian.

1. For branches with address spaces which intersect, the addresses in the intersection _must_ have distributions with the same base measure. This means you cannot swap continuous for discrete or vice versa depending on which branch you're on.

2. Mutable state does not interact well with iterative inference (e.g. MCMC). Additionally, be careful about the support of your distributions in this regard. If you're going to use mutable state in your programs, use `rand` calls in a lightweight manner - only condition on distributions with constant support and be careful about MCMC.

## Specialized call sites

Jaynes also offers a set of primitive language features for creating _specialized call sites_ which are similar to the combinators of [Gen](https://www.gen.dev/dev/ref/combinators/#Generative-Function-Combinators-1). These special features can be activated by a special set of calls (below, `markov` and `plate`).

```julia
using Jaynes

function bar(m, n)
    x = rand(:x, Normal(m, 1.0))
    q = rand(:q, Normal(x, 5.0))
    z = rand(:z, Normal(n + q, 3.0))
    return q, z
end

function foo()
    x = markov(:x, bar, 10, 0.3, 3.0)
    y = plate(:y, bar, x)
    return y
end

ret, cl = simulate(foo)
display(cl.trace; show_values = true)
```

Here, the `markov` and `plate` calls provide explicit knowledge to the tracer that the generation of randomness conforms to a computation pattern which can be vectorized. This allows the tracer to construct an efficient `VectorizedCallSite` which allows more efficient updates/regenerations than a "black-box" `CallSite` where the dependency information may not be known. This is a simple way for the user to increase the efficiency of inference algorithms, by informing the tracer of information which it can't derive on its own (at least for now 😺).

`markov` requires that the user provide a function `f` with

```math
f: (X, Y) \rightarrow (X, Y)
```

as well as a first argument which denotes the number of fold operations to compute (in the example above, `10`). `markov` will then iteratively compute the function, passing the return value as arguments to the next computation (from left to right).

`plate` does not place requirements on the function `f` (other than the implicit requirements for valid programs, as described above) but does require that the arguments be a `Vector` with each element matching the signature of `f`. `plate` then iteratively applies the function as a kernel for each element in the argument vector (similar to the functional `map` operation).

In the future, a number of other specialized call sites are planned. The fallback is always black-box tracing, but if you provide the tracer with more information about the dependency structure of your probabilistic program, it can utilize this to accelerate iterative inference algorithms. One interesting research direction is the automatic discovery of these patterns in programs: if you're interested in this, please contribute to the [open issue about automatic structure discovery](https://github.com/femtomc/Jaynes.jl/issues/31).
