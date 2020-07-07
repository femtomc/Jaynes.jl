Jaynes is a contextual _effect-oriented programming_ system for probabilistic programming. Internally, the current implementation closely follows the design of [Gen](https://www.gen.dev/) which also uses the notion of stateful execution contexts to produce the interfaces required for inference. 

In contrast to Gen (which provides powerful optimizations for programs written in the [static DSL](https://www.gen.dev/dev/ref/modeling/#Static-Modeling-Language-1)), Jaynes is focused on an optimized dynamic language which allows most Julia programs. To address the dynamic analysis problems which arise as a function of this goal, Jaynes is implemented using IR introspection and metaprogramming. The long term goal of Jaynes is to implement a set of useful optimizations _by default_ for programs expressible in the system, while providing a simple modeling and inference interface.

!!! info
    Jaynes uses many concepts from the design and implementation of Gen. First and foremost, I would recommend users of Jaynes become familiar with Gen - to understand the problems which Jaynes attempts to solve. The following papers may be useful in this regard:

    1. [Gen: a general-purpose probabilistic programming system with programmable inference](https://dl.acm.org/doi/10.1145/3314221.3314642)
    2. [Probabilistic programming with programmable inference](https://people.csail.mit.edu/rinard/paper/pldi18.pdf)
    3. [A new approach to probabilistic programming inference](http://proceedings.mlr.press/v33/wood14.pdf)
    4. [Lightweight Implementations of probabilistic programming languages via transformational compilation](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf)

    In the design space of compiler metaprogramming tools, [IRTools](https://github.com/FluxML/IRTools.jl) and [Cassette](https://github.com/jrevels/Cassette.jl) are also large influences on Jaynes. In particular, the former is a core component of the implementation.

    Jaynes has also been influenced by [Turing](https://turing.ml/dev/), the [Poutine effects system](https://docs.pyro.ai/en/stable/poutine.html) in Pyro, and [Unison lang](https://www.unisonweb.org/). Jaynes does not implement _algebraic effects_ in a rigorous (or functional!) way, but the usage of execution contexts which control how certain method calls are executed is closely aligned with these concepts.

## Implementation

Jaynes is organized around a central `IRTools` _dynamo_

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. There are a number of such inheritors

```
HierarchicalTrace
UnconstrainedGenerateContext
ConstrainedGenerateContext
ProposalContext
UpdateContext
RegenerateContext
ScoreContext
```

each of which has a special dispatch definition which allows the dynamo to dispatch on `rand` calls with addressing e.g.

```julia
@inline function (ctx::UnconstrainedGenerateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     d::Distribution{K}) where {T <: Address, K}
    s = rand(d)
    ctx.tr.chm[addr] = ChoiceSite(logpdf(d, s), s)
    return s
end
```

for `UnconstrainedGenerateContext`. Each of the other contexts define a particular functionality required for inference over probabilistic program traces. 

## Traces

The structured representation of traces is also an `ExecutionContext`

```julia
abstract type Trace <: ExecutionContext end

mutable struct HierarchicalTrace <: Trace
    chm::Dict{Address, RecordSite}
    score::Float64
    function HierarchicalTrace()
        new(Dict{Address, RecordSite}(), 0.0)
    end
end
```

so usage for unconstrained generation is simple

```julia
using Jaynes
geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)
tr = Trace()
tr(geo, 5.0)
display(tr, show_value = true)
```

will produce

```
  __________________________________

               Addresses

 flip : false
 :geo => :flip : true
  __________________________________
```

# Inference

To express constraints associated with inference (or unconstrained selections required for MCMC), there is a `Selection` interface which can be used to communicate constraints to `rand` sites in compatible execution contexts.

```julia
sel = selection((:flip, true))
ctx = Generate(Trace(), sel)
ctx(geo, 0.5)
display(ctx.tr, show_values = true)
```

will produce

```
  __________________________________

               Addresses

 flip : true
  __________________________________
```

which constrains the execution to select that value for the random choice at address `:flip`. We can also communicate constraints to inference algorithms

```julia
@time calls, lnw, lmle = Jaynes.importance_sampling(geo, (0.05, ); observations = sel)
println(lmle)
@time calls, lnw, lmle = Jaynes.importance_sampling(geo, (0.5, ); observations = sel)
println(lmle)
@time calls, lnw, lmle = Jaynes.importance_sampling(geo, (0.8, ); observations = sel)
println(lmle)
```

will produce

```julia
  0.431258 seconds (1.49 M allocations: 78.267 MiB, 4.96% gc time)
-2.9957322735539904
  0.003901 seconds (80.04 k allocations: 4.476 MiB)
-0.6931471805599454
  0.004302 seconds (80.04 k allocations: 4.476 MiB)
-0.2231435513142106
```

which shows that the log marginal likelihood of the data increases as the parameter for the geometric generator increases (towards a value which is ultimately more likely given the data).

## Black-box extensions

Jaynes is equipped with the ability to extend the tracer to arbitrary black-box code, as long as the user can provide a `logpdf` for the call

```julia
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end


cl = Jaynes.call(Trace(), rand, :geo, geo, 0.3)
display(cl.trace; show_values = true)
```

will produce

```
  __________________________________

               Addresses

 geo : 4
  __________________________________
```
