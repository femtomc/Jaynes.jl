## Universal probabilistic programming

Probabilistic programming systems are classified according to their ability to express the subset of stochastic computable functions which form valid probability distributions over program execution (in some interpretation). That's a terrible mouthful - but it's wide enough to conveniently capture systems which focus on Bayesian networks, as well as systems which capture a wider set of programs, which we will examine shortly. 

Probabilistic programming systems which restrict allowable forms of control flow or recursion are referred to as _first-order_ probabilistic programming systems. The support of the distribution over samples sites which a _first-order_ program defines can be known at compile time - this implies that these programs can be translated safely to a static graph representation (Bayesian network or factor graph) at compile time. This representation can also be attained if control flow can be _unrolled_ using compiler techniques like _constant propagation_.

A static graph representation constructed at compile time is useful, but it's not sufficient to express all valid densities over program execution. _Higher-order_ or _universal_ probabilistic programming frameworks include the ability to handle stochasticity in control flow bounds and recursion. In general, these frameworks include the ability to handle runtime sources of randomness which can't be identified at compile time. To achieve this generality, frameworks which support the ability to express these sorts of probabilistic programs are typically restricted to sampling-based inference methods.

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

One of the ways which Jaynes accomplishes this is by creating a set of "record site" abstractions which denote places where the tracer can intercept and take over for the normal execution or call semantics which the programmer expects. This notion of an interception site is central to a number of compiler plug-in style systems ([IRTools](https://github.com/FluxML/IRTools.jl) and [Cassette](https://github.com/jrevels/Cassette.jl) included). Systems like these might see a call and intercept the call, possible replacing the call with another call with extra points of overloadability. These systems do this recursively throughout the call stack (neat! 😺). As far as I know, it is rare to be able to do this natively in languages. This is a beautiful and deadly part of Julia.

To facilitate probabilistic programming, Jaynes intercepts calls to `rand` (as you might have guessed) and interprets them differently depending on the _execution context_ which the user calls on their toplevel function. The normal Julia execution context is activated by simply calling the toplevel function directly - but Jaynes provides access to a number of additional contexts which perform useful functionality for the design and implementation of sample-based inference algorithms. In general:

1. When Jaynes sees an addressed rand site `rand(:x, d)` where `d` is a `Distribution` instance from the `Distributions` package, it intercepts it and reasons about it as a `ChoiceSite` record of the interception, which may include recording some metadata to facilitate inference, or performing other operations.

2. When Jaynes sees an addressed rand site `rand(:x, fn, args...)`, it intercepts it and reasons about it as a `CallSite` record of the interception, which may include recording some metadata to facilitate inference, before then recursing into the call to find other points of interception.

These are the two basic patterns which are repeated throughout the implementation of execution contexts, which we will see in a moment.

## Implementing a context

In this section, we'll walk through the implementation of the `GenerateContext` execution context in full. This should give users of the library a good baseline understanding about how these execution contexts are implemented, and how they do what they do.

First, each context is a _dynamo_ - which is a safe `IRTools` version of Julia's [generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions-1). Generated functions have access to type information and, thus, method bodies at compile time. Generated functions are typically used to push computation to compile time, but [you can do wild things with them](https://github.com/femtomc/Mixtape.jl/blob/937068b7fd1ead7dbbc9837903cf52d0ab3a48c8/src/Mixtape.jl#L42-L57). This link showcases a generated function and IR pass which recursively wraps function calls in itself, allow you to use dispatch to intercept function calls and do whatever you want with them at any level of the call stack. This fundamental idea is how libraries like `Cassette` and the dynamos of `IRTools` do what they do - this is compiler metaprogramming at its finest (although it is currently hard on the compiler).

Dynamos are essentially better behaved generated functions. If you look at that version of `Mixtape` - it took quite a number of tries to prevent the execution engine from segfaulting out. This never happens with dynamos, with have safe fallbacks when lowered method meta information cannot be acquired. There are also a number of convenient benefits to working with `IRTools` - the SSA IR format provided by the library is very nice to work with, and there are a number of utilities for compiler enthusiasts to use when writing custom IR passes. For our implementation, we don't need of these advanced utilities - we will start with a simple dynamo:

```julia
abstract type ExecutionContext end

@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

This defines a _closure_ - a callable object - as a dynamo (which, remember, is basically an easy-to-work-with generated function). What does this dynamo do? First, it grabs lowered metadata for a particular call, then it converts this metadata to `IRTools` IR with the call to `IR(a...)`. If the dynamo can't perform this first part of the process, the `IR` will safely be an instance of `Nothing`, so the dynamo will return `nothing`, which means that it just calls the original call with args. If you can derive IR for the call, you pass the IR into `recur!` which performs the sort of recursive wrapping from `Mixtape` in a safe way, so that the dynamo wraps every call down the stack (`recur!` is a `Jaynes` specific version of `recurse!` from `IRTools` which includes a few optimizations specific to `Jaynes`).

The end result of this definition is: if you define any concrete struct which inherits from `ExecutionContext`, you can call it on a function type and args, and it will wrap itself around every call in the resultant call stack - which means that, as the function call executes, any call on the branch you are on gets wrapped as well, and the transformation repeats itself, until it hits primitives for which it can't derived lowered metadata (and it will just call those primitives, instead of wrapping).

What does this afford us? Well, we can now define through dispatch the behavior for any function call we want, for any inheritor of `ExecutionContext`:

```julia
mutable struct GenerateContext{T <: Trace, 
                               K <: ConstrainedSelection, 
                               P <: Parameters} <: ExecutionContext
    tr::T
    select::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
end

@inline function (ctx::GenerateContext)(call::typeof(rand), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    if has_query(ctx.select, addr)
        s = get_query(ctx.select, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, ChoiceSite(score, s))
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, ChoiceSite(logpdf(d, s), s))
    end
    return s
end
```

Now, we define a concrete inheritor of `ExecutionContext` called `GenerateContext` which keeps a few pieces of metadata around which we will use to record information about calls which include random choices. The inlined closure definition below the struct definition outlines what happens when the dynamo wrapping encounters a call of the following form:

```julia
rand(addr::T, d::Distribution{K}) where T <: Address
```

where `Address` is a `Union{Symbol, Pair{Symbol, Int}}` and is used by the user to denote the sites in their probabilistic program which the tracer will pay attention to. What happens in this call instead of the normal execution for `rand(addr, d)`? First we do some bookkeeping to make sure the probabilistic program is valid using `visit!`, then we check a field called `select` to determine if the user has provided any constraints (i.e. observations) which the execution context should use to constrain this call at this address. If we do have a constraint, we grab the constraint, score it using `logpdf` for the distribution in the call and add a record of the call to a piece of metadata called a `Trace` in the execution context. Otherwise, we randomly sample and record the call in the `Trace`. Finally, we return the sample (or observation) `s`.

This is exactly what happens in the `GenerateContext` every time the dynamo sees a call of the `rand` form above instead of the normal execution. But this is exactly what we need to allow sampling of probabilistic programs where some of the address have user-provided constraints. And it all happens automatically, courtesy of compiler metaprogramming.

The other execution contexts are implemented in the same way - you'll also notice that this implementation is repeated in the set of _specialized call sites_ which the user can activate if they'd like to express part of a probabilistic program which confirms to a certain structure of randomness dependency. As long as the required interception occurs at the function call level, this compiler metaprogramming technique can be used. Very powerful!
