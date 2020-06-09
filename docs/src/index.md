There are many active probabilistic programming frameworks in the Julia ecosystem (see [Related Work](related_work.md)) - the ecosystem is one of the richest sources of probabilistic programming research in any language. Frameworks tend to differentiate themselves based upon what model class they efficiently express ([Stheno](https://github.com/willtebbutt/Stheno.jl) for example allows for convenient expression of Gaussian processes). Other frameworks support universal probabilistic programming with sample-based methods, and have optimized features which allow the efficient composition/expression of inference queries (e.g. [Turing](https://turing.ml/dev/) and [Gen](https://github.com/probcomp/Gen.jl)). Jaynes sits within this latter camp - it is strongly influenced by Turing and Gen, but more closely resembles a system like [Zygote](https://github.com/FluxML/Zygote.jl). The full-scope Jaynes system will allow you to express the same things you might express in these other systems - but the long term research goals may deviate slightly from these other libraries. In this section, I will discuss a few of the long term goals.

---

### Graphical model DSL

One of the research goals of Jaynes is to identify _composable interfaces_ for allowing users to express static graphical models alongside dynamic sample-based models. This has previously been a difficult challenge - the representations which each class of probabilistic programming system utilizes is very different. Universal probabilistic programming systems have typically relied on sample-based inference, where the main representation is a structured form of an execution trace. In contrast, graphical model systems reason explicitly about distributions and thus require an explicit graph representation of how random variates depend on one another.

A priori, there is no reason why these representations can't be combined in some way. The difficulty lies in deciding how to switch between representations when a program is amenable to both, as well as how the different representations will communicate across inference interfaces. For example, consider performing belief propagation on a model which supports both discrete distributions and function call sites for probabilistic programs which required a sample-based tracing mechanism for interpretation. To enable inference routines to operate on this "call graph" style representation, we have to construct and reason about the representation separately from the runtime of each program.

### Density compilation

TODO.

### Automatic inference compilation

Jaynes already provides (rudimentary) support for gradient-based learning in probabilistic programs. Jaynes also provides a simple interface to construct and use [_inference compilers_](https://arxiv.org/abs/1610.09900). The library function `inference_compilation` provides access to the inference compiler context. The result of inference compilation is a trained inference compiler context which can be used to generate traces for the posterior conditioned on observations.

```julia
function foo1()
    x = rand(:x, Normal, (3.0, 10.0))
    y = rand(:y, Normal, (x + 15.0, 0.3))
    return y
end

ctx = inference_compilation(foo1, (), :y; batch_size = 256, epochs = 100)
obs = constraints([(:y, 10.0)])
ctx, tr, score = trace(ctx, obs)
```

The user must provide a target observation address to the `inference_compilation` call. This allows the inference compiler to construct an observation-specific head during training.

This inference method is not yet fully tested - but you can take a peek in the `src` to see how it will eventually be stabilized. One of the long term goals of Jaynes is to provide a backend for inference compilation of arbitary programs. If a user does not specify the choice map structure of the program, the addresses will be automatically filled in, with enough reference metadata to allow the user to locate the `rand` call in the original program. Of course, it is always preferable to structure your own choice map space - this feature is intended to allow programs with untraced `rand` calls to utilize a useful (but possibly limited) form of inference.

### Black-box extensions

_Jaynes_ is equipped with the ability to extend the tracing interface to black-box code. This is naturally facilitated by the metaprogramming capabilities of `Cassette`. The primary usage of this extension is to define new `logpdf` method definitions for code which may contain sources of randomness which are not annotated with addresses and/or where inspection by the tracing mechanism can be safely abstracted over. Thus, `@primitive` defines a contract between the user and the tracer - we assume that what you're doing is correct and we're not going to check you on it!

The following example shows how this extension mechanism works.

```julia
function foo(y::Float64)
    # Untraced randomness.
    y = rand(Normal(0.5, 3.0))
    return y
end

@primitive function logpdf(fn::typeof(foo), args::Tuple{Float64}, y::Float64)
    if y < 1.0
        log(1) 
    else
        -Inf
    end
end

function bar(z::Float64)
    y = rand(:y, foo, (z, ))
    return y
end

ctx = Generate(Trace())
ret = trace(ctx, bar, (0.3, ))
println(ctx.metadata.tr)

#  __________________________________
#
#               Playback
#
# y
#          val  = 2.8607525733342767
#
#  __________________________________
#
# score : 0.0
#
#  __________________________________

```

`@primitive` requires that the user define a `logpdf` definition for the call. This expands into `overdub` method definitions for the tracer which automatically work with all the core library context/metadata dispatch. The signature for `logpdf` should match the following type specification:

```julia
logpdf(::typeof(your_func), ::Tuple, ::T)
```

where `T` is the return type of `your_func`. 

Note that, if your defined `logpdf` is differentiable - gradients will automatically be derived for use in `Gradient` learning contexts as long as `Zygote` can differentiate through it. This can be used to e.g. train neural networks in `Gradient` contexts where the loss is wrapped in the `logpdf`/`@primitive` interface mechanism.

The extension mechanism _does not_ check if the user-defined `logpdf` is valid. This mechanism also overrides the normal fallback (i.e. tracing into calls) for any function for which the mechanism is used to write a `logpdf` - this means that if you write a `logpdf` using this mechanism for a call and there _is_ addressed randomness in the call, it will be ignored by the tracer.

---

### Summary 

To facilitate these research goals, Jaynes is designed as a type of compiler plugin. In contrast to existing frameworks, Jaynes does not require the use of specialized macros to denote where the modeling language begins and ends. The use of macros to denote a language barrier has a number of positive advantages from a user-facing perspective, but some disadvantages related to composability. As an opinion, I believe that a general framework for expressing probabilistic programs should mimic the philosophy of _differentiable programming_. The compiler plugin backend should prevent users from writing programs which are "not valid" (either as a static analysis or a runtime error) but should otherwise get out of the way of the user. Any macros present in the Jaynes library extend the core functionality or provide convenient access to code generation for use by a user - but are not required for modeling and inference.

Because Jaynes is a compiler plugin, it is highly configurable. The goal of the core package is to implement a set of "sensible defaults" for common use, while allowing the implementation of other DSLs, custom inference algorithms, custom representations, etc on top. In this philosophy, Jaynes follows a path first laid out by Gen and Zygote...with a few twists.

> Bon appétit!
