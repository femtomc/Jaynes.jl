# Hitchhiker's guide to Jaynes

This is a small guide which details the layout of the repository.

The core data structures live in `/core` - these data structures encompass trace representations, call site representations, and address map representations. This directory also includes the `target` interface (e.g. the thing you use to post observations to your model functions) and the `selection` interface (e.g. the thing you use to ask certain contexts to `regenerate` choices for a sample trace).

All the execution contexts live in `/contexts`.

The compiler directory `/compiler` is a highly experimental part of the codebase which deals with dynamically optimizing function calls when used in contexts.

The language extension directory `/language_extensions` handles sugar macros, defining primitive calls for the tracers, as well as interfaces across languages and within Julia PPLs.

The inference directory `/inference` provides implementations of a number of standard inference building blocks - including a Metropolis-Hastings operator, importance sampling, particle filtering operators, variational inference operators, maximum likelihood and maximum a posterior optimization. Included in this directory are also a number of MCMC kernels (e.g. Hamiltonian Monte Carlo, elliptical slice, a custom piecewise deterministic Markov kernel which is a work in progress).
