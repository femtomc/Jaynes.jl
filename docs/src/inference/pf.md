```@meta
CurrentModule = Jaynes
```

```@docs
initialize_filter
filter_step!
```

!!! info
    Custom proposals provided to `filter_step!` should accept _as first argument_ a `CallSite` instance (e.g. so that you can use the previous trace and information in your transition proposal).

```@docs
check_ess_resample!
```
