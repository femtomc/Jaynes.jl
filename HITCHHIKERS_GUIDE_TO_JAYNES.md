# Hitchhiker's guide to Jaynes

This is a small guide which details the layout of the repository.

The core data structures live in `/core` - these data structures encompass trace representations, call site representations, and address map representations. This directory also includes the `target` interface (e.g. the thing you use to post observations to your model functions) and the `selection` interface (e.g. the thing you use to ask certain contexts to `regenerate` choices for a sample trace).

All the execution contexts and compilation pipelines live in `/pipelines`.

The compiler directory `/compiler` is a highly experimental part of the codebase which deals with ingredients for compiler pipelines/the execution of models when used in contexts. This is highly likely to change in highly variable ways 😸. The compiler directory also deals with other experimental parts of the system - like compiling generic stochastic simulators to probabilistic programs, or type inference on IR representations of generative functions.

The macros directory `/macros` handles sugar macros and defining primitive calls for the tracers.
