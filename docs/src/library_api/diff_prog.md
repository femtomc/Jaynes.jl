```@meta
CurrentModule = Jaynes
```

Jaynes supports Zygote-based reverse mode gradient computation of learnable parameters and primitive probabilistic choices. This functionality is accessed through two different gradient contexts.

```@docs
ParameterBackpropagateContext
```

For one-shot gradient computations, this context is easily accessed through the `get_parameter_gradients` method.

```@docs
get_parameter_gradients
```

For choices, the context is `ChoiceBackpropagateContext`.

```@docs
ChoiceBackpropagateContext
```

For one-shot gradient computations on choices, the `ChoiceBackpropagateContext` is easily accessed through the `get_choice_gradients` method.

```@docs
get_choice_gradients
```

In the future, Jaynes will support a context which allows the automatic training of neural network components (`Flux.jl` or otherwise) facilitated by custom call sites. [See the foreign model interface for more details](fmi.md).

## Updating parameters

As part of standard usage, it's likely that you'd like to update learnable parameters in your model (which you declare with `learnable(addr, initial_value)`. 

!!! warning
    Currently, there's a bug with declaring learnable structures which prevents the usage of parameters of type other than `Float64` or `Array{Float64, 1}`. If you run into Zygote errors with mutating arrays, you can try to alleviate the problem by annotating your parameters (e.g. `Float64[1.0, 3.0, ...]`). You'll mostly be okay if you can stick to scalar parameters and 1D arrays - I'm working to identify this issue and fix it so higher-rank tensors can also be used.

Given a `CallSite`, you can extract parameters using `get_parameters`

```julia
ret, cl, w = generate(selection((:q, -0.5)), learnable_hypers)
params = get_parameters(cl)
```

which produces an instance of `LearnableParameters` by unpacking the `params` field in any trace kept in the call site. `LearnableParameters` and `Gradients` are explicitly kept separate from the internal call site representation, to encourage portability, as well as non-standard optimization schemes. Jaynes links up with the API provided by `Flux.Optimisers` through `update!` - this allows you to use any of the optimisers provided by `Flux` to update your parameters. Given a `Gradients` instance, to update your parameters, just call `update_parameters` with your favorite optimiser

```julia
trained_params = update_parameters(opt, params, gradients)
```

which produces a new instance of `LearnableParameters` after apply the gradient descent step. You can then pass these in as an argument to the normal context interfaces (e.g. `generate`, `simulate`, etc) to use your updated parameters.
