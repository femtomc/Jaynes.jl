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
