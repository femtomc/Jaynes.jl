@def title = "Implementation architecture"

Jaynes is organized around a central [IRTools](https://github.com/FluxML/IRTools.jl) [dynamo](https://fluxml.ai/IRTools.jl/latest/dynamo/):

```julia
@dynamo function (mx::ExecutionContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end
```

which defines how instances of inheritors of `ExecutionContext` act on function calls. The "wrapping transform" implemented through `recur!` is customized, to make the tracer lightweight and to prevent some type stability issues from calls to `Base` Julia.

```julia
unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

whitelist = [
             # Base.
             :rand, :_apply_iterate, :collect,

             # Specialized call sites.
             :markov, :plate, :cond, 

             # Interactions with the context.
             :learnable, :fillable, :factor,

             # Foreign model interfaces.
             :foreign, :deep
            ]

# Fix for specialized tracing.
function recur!(ir, to = self)
    for (x, st) in ir
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || continue
            ir[x] = Expr(:call, to, st.expr.args...)
        end
    end
    return ir
end
```

so we see that the tracer is only allowed to look at certain calls, and uses a few fixes for some common issues. This drastically improves the performance over a "heavyweight" tracer which looks at everything. For the use case of probabilistic programming [implemented in this style](http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf), it's perfectly acceptable.

There are a number of inheritors for `ExecutionContext`

```
GenerateContext
SimulateContext
ProposalContext
UpdateContext
RegenerateContext
ScoreContext
ChoiceBackpropagateContext
ParameterBackpropagateContext
```

each of which has a set of special dispatch definition which allows the dynamo to dispatch on `rand` calls with user-provided addressing. As an example, here's the interception dispatch inside the `GenerateContext` (which we just examined in the last section):

```julia
@inline function (ctx::GenerateContext)(call::typeof(rand), 
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
* `score`

(roughly, `Gen.jl` may change their interfaces, these may also change here - but the ideas behind these interfaces will remain the same)

You use these high-level interface calls on your models e.g.

```julia
ret, cl = simulate(some_model, args...)
```

which takes care of constructing a `SimulateContext`, executing your model with `args...` in that context, and bundling up the return and a record of that call for you.
