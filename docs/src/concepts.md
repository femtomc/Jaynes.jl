The majority of the concepts used in the initial implementation of this package come from a combination of research papers and research systems (the most notable in the Julia ecosystem is [Gen](https://www.gen.dev/)). See [Related Work](related_work.md) for a comprehensive list of references.

## Universal probabilistic programming

Probabilistic programming systems are classified according to their ability to express stochastic computable functions. In particular, languages which restrict allowable forms of control flow or recursion are referred to as _first-order_ probabilistic programming languages. The support of the distribution over samples sites which a _first-order_ program defines can be known at compile time - this implies that these programs can be translated safely to a static graph representation (a Bayesian network). This representation can also be attained if control flow can be _unrolled_ using compiler techniques like _constant propagation_.

A static graph representation is useful, but it's not sufficient to express all stochastic computable functions. _Higher-order_ or _universal_ probabilistic programming frameworks include the ability to handle stochasticity in control flow bounds and recursion. To achieve this generality, frameworks which support the ability to express these sorts of probabilistic programs are typically restricted to sampling-based inference methods (which essentially reflects the duality between a _compiler-based approach_ to model representation and an _interpreter-based approach_ which requires a number of things to be determined at runtime). Modern systems blur the line between these approaches (see [Gen's static DSL](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1) for example).

## The choice map abstraction

One important concept in the universal space is the idea of a map from call sites where random choices occur to the values at those sites. This map is called a _choice map_ in most implementations (original representation in [Bher](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)). The semantic interpretation of a probabilistic program expressed in a framework which supports universal probabilistic programming via the choice map abstraction is a distribution over choice maps. Consider the following program, which expresses the geometric distribution in this framework:

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

One simple question arises: what exactly does this _distribution over choice maps_ look like in a mathematical sense? The distribution looks something like this:

$P(x)$


## Programming the distribution over choice maps

When interacting with a probabilistic programming framework which utilizes the choice map abstraction, the programming model requires that the user keep unique properties of the desired distribution over choice maps in mind. Here are a set of key difference between classical parametric Bayesian models and universal probabilistic programs:

## Inference
