This is a set of compile-time (and, occasionally, runtime) tools for specialization, type checking, etc. There is not a main `compile` entry point - instead these tools are used in a variety of places. You'll typically find the usage of these tools in the `pipeline` sub-directory, as well as in `JFunction` construction for static support checking/type checking.

At generative function construction time - `absint` provides a set of capabilities which allows Jaynes to check for support errors and derive the trace type of functions which satisfy a restricted DSL.

At generated function expansion time (e.g. execution context calls) - `specializer` provides a primitive type system which allows the propagation of compile-time argument difference type information. This system is used to prune calls which are irrelevant to the inference semantics of `update` and `regenerate`. `specializer` is built on top of the functionality provided by `absint`.

`jaynesizer` is slightly different than the above two (which focus on verifying and specializing code written "for Jaynes"). `jaynesizer` transforms black box code with randomness into the dynamic DSL format for Jaynes. It also operates at generated function expansion time. `jaynesizer` shares much of its codebase with `Genify` - in the future, these two will likely share a common infrastructure.
