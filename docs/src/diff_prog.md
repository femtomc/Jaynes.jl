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

As part of standard usage, it's likely that you'd like to update learnable parameters in your model (which you declare with `learnable(addr, initial_value)`. Given a `CallSite`, you can extract parameters using `get_parameters`

```julia
ret, cl, w = generate(selection((:q, -0.5)), learnable_hypers)
params = get_parameters(cl)
```

which produces an instance of `LearnableParameters` by unpacking the `params` field in any trace kept in the call site. `LearnableParameters` and `Gradients` are explicitly kept separate from the internal call site representation, to encourage portability, as well as non-standard optimization schemes. Given a `Gradients` instance, to update your parameters, just call `update!`

```julia
update!(params, gradients)
```

which mutates the parameters in-place. You can then pass these in as an argument to the normal context interfaces (e.g. `generate`, `simulate`, etc) to use your updated parameters.
