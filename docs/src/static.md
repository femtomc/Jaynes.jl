One of the powerful benefits of the compiler plugin approach is the ability to derive additional analysis information, which can be used to structure the `Trace` representation of the program. 

Jaynes features a set of `Trace` representations which "fill in" the call structure of programs which users are likely to write. The most general representation is `HierarchicalTrace` which features no metadata about the program. This is the default `Trace` representation for calls where we cannot determine the exact address space at compile time (e.g. loops require runtime information, or rely on randomness). `VectorizedTrace` is a representation which is used by `Jaynes` _effects_ (which are semantically similar to the combinators of `Gen`). These effects amount to annotations by the programmer that the randomness flow satisfies a set of constraints which allows the construction of a specialized `Trace` form.

The most performant `Trace` representation requires a compile-time call graph analysis which we elaborate below. This analysis is performed _by default_ for all model programs which the user writes. This analysis is utilized by the tracing system at runtime to construct an optimal `Trace` representation which is a combination of the above trace types.

### GraphTrace

Jaynes utilizes a particular transformation to derive an optimal representation for the `Trace`. The resultant `Analysis` representation is used at runtime to construct optimal trace representations for each call. Below, we outline the static pass, and briefly cover how the specialized `GraphTrace` can be used to cache computations, and identify when calls do not need to be re-computed.
