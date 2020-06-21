The majority of the concepts used in the initial implementation of this package come from a combination of research papers and research systems (the most notable in the Julia ecosystem is [Gen](https://www.gen.dev/)). See [Related Work](related_work.md) for a comprehensive list of references.

## Universal probabilistic programming

Probabilistic programming systems are classified according to their ability to express the subset of stochastic computable functions which form valid probability densities over program execution (in some interpretation). That's a terrible mouthful - but it's wide enough to conveniently capture systems which focus on Bayesian networks, as well assystems which capture a wider set of programs, which we will examine shortly. 

Probabilistic programming systems which restrict allowable forms of control flow or recursion are referred to as _first-order_ probabilistic programming systems. The support of the distribution over samples sites which a _first-order_ program defines can be known at compile time - this implies that these programs can be translated safely to a static graph representation (a Bayesian network). This representation can also be attained if control flow can be _unrolled_ using compiler techniques like _constant propagation_.

A static graph representation is useful, but it's not sufficient to express all valid densities over program execution. _Higher-order_ or _universal_ probabilistic programming frameworks include the ability to handle stochasticity in control flow bounds and recursion. To achieve this generality, frameworks which support the ability to express these sorts of probabilistic programs are typically restricted to sampling-based inference methods (which reflects a duality between a _compiler-based approach_ to model representation and an _interpreter-based approach_ which requires a number of things to be determined at runtime). Modern systems blur the line between these approaches (see [Gen's static DSL](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1) for example) when analysis or annotation can improve inference performance.

## The choice map abstraction

One important concept in the universal space is the notion of a mapping from call sites where random choices occur to the values at those sites. This map is called a _choice map_ in most implementations (original representation in [Bher](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)). The semantic interpretation of a probabilistic program expressed in a framework which supports universal probabilistic programming via the choice map abstraction is a distribution over choice maps. Consider the following program, which expresses the geometric distribution in this framework:

```julia
geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)
```

Here, `rand` call sites are also given addresses and recursive calls produce a hierarchical address space. A sample from the distribution over choice maps for this program might produce the following map:

```julia
 :geo => :flip
          val  = false

 flip
          val  = false

 :geo => (:geo => :flip)
          val  = false

 :geo => (:geo => (:geo => :flip))
          val  = false

 :geo => (:geo => (:geo => (:geo => :flip)))
          val  = true
```

One simple question arises: what exactly does this _distribution over choice maps_ look like in a mathematical sense? To answer this question, we have to ask how control flow and iteration language features affect the "abstract space" of the shape of the program trace. For the moment, we will consider only randomness which occurs explicitly at addresses in each method call (i.e. `rand` calls with distributions as target) - it turns out that we can safely focus on the shape of the trace in this case without loss of generalization. Randomness which occurs inside of a `rand` call where the target of the call is another method call can be handled by the same techniques we introduce to analyze the shape of a single method body without target calls. Let's make this concrete by analyzing the following program:

```julia
function foo(x::Float64)
    y = rand(:y, Normal, (x, 1.0))
    if y > 1.0
        z = rand(:z, Normal, (y, 1.0))
    end
    return y
end
```

So, there are basically two "branches" in the trace. One branch produces a distribution over the values of the addressed random variables

```math
\begin{equation}
P(y, z; x) = P(z | y)P(y | x)
\end{equation}
```

and the other branch produces a distribution:

```math
\begin{equation}
P(y ; x) = P(y | x)
\end{equation}
```

but how do you express this as a valid measure _itself_? One way is to think about an auxiliary "indicator" measure over the possible sets of _addresses_ in the program:

```math
\begin{equation}
P(A), \text{where $A$ takes a set as a value}
\end{equation}
```

We actually know a bit about the space of values of ``A`` and about this measure.  

1. The space of values is the powerset of the set of all addresses.
2. The measure ``P(A)`` is unknown, but if the program halts, it is normalized.

So we can imagine that the generative process selects an ``A``

## Programming the distribution over choice maps

When interacting with a probabilistic programming framework which utilizes the choice map abstraction, the programming model requires that the user keep unique properties of the desired distribution over choice maps in mind. Here are a set of key difference between classical parametric Bayesian models and universal probabilistic programs:

## Inference
