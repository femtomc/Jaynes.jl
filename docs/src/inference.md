```@meta
CurrentModule = Jaynes
```

```@docs
importance_sampling
```

!!! info
    Addressed randomness in a custom proposal should satisfy the following criteria to ensure that inference is mathematically valid:
    1. Custom proposals should only propose to unobserved addresses in the original program.
    2. Custom proposals should not propose to addresses which do not occur in the original program.
