```@meta
CurrentModule = Jaynes
```

## Importance sampling

```@docs
importance_sampling
```

!!! info
    Addressed randomness in a custom proposal passed to `importance_sampling` should satisfy the following criteria to ensure that inference is mathematically valid:
    1. Custom proposals should only propose to unobserved addresses in the original program.
    2. Custom proposals should not propose to addresses which do not occur in the original program.

## Particle filtering

```@docs
initialize_filter
filter_step!
```

!!! info
    Custom proposals provided to `filter_step!` should accept _as first argument_ a `CallSite` instance (e.g. so that you can use the previous trace and information in your transition proposal).

```@docs
resample!
```

`resample!` can be applied to both instances of `Particles` produced by particle filtering, as well as instances of `Particles` produced by importance sampling.

## Variational inference

```@docs
advi
```

## Markov chain Monte Carlo

```@docs
metropolis_hastings
```

!!! info
    Similar to custom proposals for particle filtering, a custom proposal passed to `metropolis_hastings` should accept an instance of `CallSite` as the first argument, followed by the other proposal arguments.

    Additionally, if the proposal proposes to addresses which modify the control flow of the original call, it must also provide proposal choice sites for any addressed which exist on that branch of the original model. If this criterion is not satisfied, the kernel is not stationary with respect to the target posterior of the model.
