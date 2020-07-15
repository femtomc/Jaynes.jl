```@meta
CurrentModule = Jaynes
```

```@docs
metropolis_hastings
```

!!! info
    Similar to custom proposals for particle filtering, a custom proposal passed to `metropolis_hastings` should accept an instance of `CallSite` as the first argument, followed by the other proposal arguments.

    Additionally, if the proposal proposes to addresses which modify the control flow of the original call, it must also provide proposal choice sites for any addressed which exist on that branch of the original model. If this criterion is not satisfied, the kernel is not stationary with respect to the target posterior of the model.

