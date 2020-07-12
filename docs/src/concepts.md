The majority of the concepts used in the initial implementation of this package come from a combination of research papers and research systems (the most notable in the Julia ecosystem is [Gen](https://www.gen.dev/)). See [Related Work](related_work.md) for a more comprehensive list of references.

## Universal probabilistic programming

Probabilistic programming systems are classified according to their ability to express the subset of stochastic computable functions which form valid probability distributions over program execution (in some interpretation). That's a terrible mouthful - but it's wide enough to conveniently capture systems which focus on Bayesian networks, as well as systems which capture a wider set of programs, which we will examine shortly. 

Probabilistic programming systems which restrict allowable forms of control flow or recursion are referred to as _first-order_ probabilistic programming systems. The support of the distribution over samples sites which a _first-order_ program defines can be known at compile time - this implies that these programs can be translated safely to a static graph representation (a Bayesian network). This representation can also be attained if control flow can be _unrolled_ using compiler techniques like _constant propagation_.

A static graph representation constructed at compile time is useful, but it's not sufficient to express all valid densities over program execution. _Higher-order_ or _universal_ probabilistic programming frameworks include the ability to handle stochasticity in control flow bounds and recursion. In general, these frameworks include the ability to handle runtime sources of randomness which can't be identified at compile time. To achieve this generality, frameworks which support the ability to express these sorts of probabilistic programs are typically restricted to sampling-based inference methods. Here, we first get a glimpse of the (well-known) duality between a _compiler-based approach_ to model representation and _interpreter-based approaches_ which allow for random computation to be determined at runtime). Modern systems blur the line between these approaches (see [Gen's static DSL](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1) for example) when analysis or annotation can improve inference performance.

## The choice map abstraction

One important concept in the universal space is the notion of a mapping from call sites where random choices occur to the values at those sites. This map is called a _choice map_ in most implementations (original representation in [Bher](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)). The semantic interpretation of a probabilistic program expressed in a framework which supports universal probabilistic programming via the choice map abstraction is a distribution over choice maps. Consider the following program, which expresses the geometric distribution in this framework:

```julia
geo(p::Float64) = rand(:flip, Bernoulli, (p, )) == 1 ? 0 : 1 + rand(:geo, geo, p)
```

Here, `rand` call sites are also given addresses and recursive calls produce a hierarchical address space. A sample from the distribution over choice maps for this program might produce the following map:

```julia
 :geo => :flip : false
 flip : false
 :geo => (:geo => :flip) : false
 :geo => (:geo => (:geo => :flip)) : false
 :geo => (:geo => (:geo => (:geo => :flip))) : true
```

One simple question arises: what exactly does this _distribution over choice maps_ look like in a mathematical sense? To answer this question, we have to ask how control flow and iteration language features affect the "abstract space" of the shape of the program trace. For the moment, we will consider only randomness which occurs explicitly at addresses in each method call (i.e. `rand` calls with distributions as target) - it turns out that we can safely focus on the shape of the trace in this case without loss of generalization. Randomness which occurs inside of a `rand` call where the target of the call is another method call can be handled by the same techniques we introduce to analyze the shape of a single method body without target calls.

## Choice and call site abstractions

Ideally, we'd like the construction of probabilistic programs to parallel the construction of regular programs - we'd like the additional probabilistic semantics to leave the original execution semantics invariant (mostly). In other words, we don't want to give up the powerful abstractions and features which we have become accustomed to while programming in Julia normally. Well, there's good news - you don't have to! You will have to keep a few new things in mind (see [the modeling language section](modeling_lang.md) for more details) but the whole language should remain open for your use.

One of the ways which Jaynes accomplishes this is by creating a set of "record site" abstractions which denote places where the tracer can intercept and take over for the normal execution or call semantics which the programmer expects. This notion of an interception site is central to a number of compiler plug-in style systems (`IRTools` and `Cassette` included). Systems like these might see a call and intercept the call, possible replacing the call with another call with extra points of overloadability. Oh, I should also mention that these systems do this recursively through the call stack 😺. As far as I know, it is rare to be able to do this natively in languages. You definitely need your language to be dynamic and likely JIT compiled (so that you can access parts of the intermediate representation) - in other words, Julia.

To facilitate probabilistic programming, Jaynes intercepts calls to `rand` (as you might have guessed) and interprets them differently depending on the _execution context_ which the user calls on their toplevel function. The normal Julia execution context is activated by simply calling the toplevel function directly - but Jaynes provides access to a number of additional contexts which perform useful functionality for the design and implementation of sample-based inference algorithms. In general:

1. When Jaynes sees an addressed rand site `rand(:x, d)` where `d` is a `Distribution` instance from the `Distributions.jl` package, it intercepts it and reasons about it as a `ChoiceSite` record of the interception, which may include recording some metadata to facilitate inference, or performing other operations.

2. When Jaynes sees an addressed rand site `rand(:x, fn, args...)`, it intercepts it and reasons about it as a `CallSite` record of the interception, which may include recording some metadata to facilitate inference, before then recursing into the call to find other points of interception.

These are the two basic patterns which are repeated throughout the implementation of execution contexts, which we will see in a moment.

## Implementing a context

In this section, we'll walk through the implementation of the `Generate` context in full. This should give users of the library a good baseline understand about how these contexts are setup, and how they do what they do.
