@def title = "Implementation architecture"

> This section assumes some level of understanding of [Julia](https://julialang.org/) as well as the [various](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions) [flavors](https://github.com/jrevels/Cassette.jl) of [staged programming](https://fluxml.ai/IRTools.jl/latest/dynamo/) in Julia.

Jaynes implements [the generative function interface](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1) for Julia functions. To enable these modelling capabilities for normal Julia functions, Jaynes is organized around a set of [IRTools](https://github.com/FluxML/IRTools.jl) [dynamos](https://fluxml.ai/IRTools.jl/latest/dynamo/). Here's a rough schema (each inheritor of `ExecutionContext` may have additional specialized transformations available to it):

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. [For those who are unfamiliar with dynamos, the call to `recur!` wraps each function call in the IR representation of the method which the context is applied to with itself.](https://github.com/femtomc/Mixtape.jl) The "wrapping transform" implemented through `recur!` is customized to ignore certain calls in `Base` and `Core` Julia - with the purpose of making the tracer lightweight, as well as preventing some type stability issues.

> As of `v0.1.28`, the above is slightly untrue - as Jaynes will now automatically address un-addressed sources of randomness in programs. This requires that it recurses into more calls than the initial version - but it still ignores a large set of primitive calls in `Base` and `Core`. In "idiomatic" modelling code, the user is [almost surely](https://en.wikipedia.org/wiki/Almost_surely) safe.

There are a number of inheritors for `ExecutionContext` - each generative function interface method gets a context:

```
GenerateContext
SimulateContext
ProposalContext
UpdateContext
RegenerateContext
AssessContext
ChoiceBackpropagateContext
ParameterBackpropagateContext
```

Each context has a special dispatch definition which allows the dynamo which defines the context to dispatch on `trace` calls with user-provided addressing. As an example, here's the interception dispatch inside the `GenerateContext` (which we just examined in the last section):

```julia
@inline function (ctx::GenerateContext)(call::typeof(trace), 
                                        addr::T, 
                                        d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    if has_value(ctx.target, addr)
        s = getindex(ctx.target, addr)
        score = logpdf(d, s)
        add_choice!(ctx, addr, score, s)
        increment!(ctx, score)
    else
        s = rand(d)
        add_choice!(ctx, addr, logpdf(d, s), s)
    end
    return s
end
```

so this context records the random choice, as well as performs some bookkeeping with the `logpdf` which we will use for inference programming. Each of the other contexts define unique interception dispatch to implement functionality required for inference over probabilistic program traces. [These can be found here.](https://github.com/femtomc/Jaynes.jl/tree/master/src/contexts)

## Sugar

The programmer is not expected to interact with these contexts directly. Instead, the programmer can utilize a set of high-level function calls which construct the contexts, run a function call with some arguments in the context, and return useful information (usually, a return value, a bundled record of the call in a `CallSite` instance, and some other probabilistic metadata). These high-level calls match the same high-level calls in `Gen.jl`:

* `simulate`
* `propose`
* `generate`
* `update`
* `regenerate`
* `assess`

(roughly, `Gen.jl` may change their interfaces, these may also change here - but the ideas behind these interfaces will remain the same)

If you so choose, you may use these high-level interface calls directly on your model functions e.g.

```julia
ret, cl = simulate(some_model, args...)
```

which takes care of constructing a `SimulateContext`, executing your model with `args...` in that context, and bundling up the return and a record of that call for you.

## Inference

Generically, if you're hoping to perform inference, you'll use the APIs from [Gen.jl](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1) to do that. The direct calls described above are used to implement these APIs. Of course, this means that programs you write using the DSL provided by Jaynes *should* be compatible with any inference algorithms you express using the Gen APIs.

In practice, Jaynes has been tested with examples for the following inference algorithms from the inference library of `Gen.jl`:

- Black box variational inference
- Importance sampling
- Involution MCMC
- Custom kernels synthesized with the [Composite Kernel DSL](https://www.gen.dev/dev/ref/mcmc/#Composite-Kernel-DSL-1)
- Metropolis-Hastings with custom proposals

[Examples of usage are available in the examples directory.](https://github.com/femtomc/Jaynes.jl/tree/master/examples)
