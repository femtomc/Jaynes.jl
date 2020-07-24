Jaynes features an extensive `Selection` query language for addressing sources of randomness. The ability to constrain random choices, compute proposals for random choices in MCMC kernels, as well as gradients requires a solid set of interfaces for selecting addresses in `rand` calls in your program. Here, we present the main interfaces which you are likely to use. This set of interfaces specifies a sort of _query language_ so you'll find common operations like union, intersection, etc which you can use to flexibly combined selection queries for use in your inference programs and modeling.

## Basic selections

These are the basic set of `Selection` APIs which allow the user to query and observe throughout the call stack of the program.

```@docs
selection
anywhere
```

## Compositions of selections

The selections produced by the above APIs can be combined compositionally to form more complex selections. Some of these compositions are only available for subtypes of `UnconstrainedSelection`.

## Selection utilities

There are a number of useful utility functions defined on subtypes of `Selection`. Many of these utilities are used internally - here are a few which may be of interest to the user.
