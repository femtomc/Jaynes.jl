```@meta
CurrentModule = Jaynes
```

Due to the design and implementation as an IR metaprogramming tool, Jaynes sits at a slightly privileged place in the probabilistic programming ecosystem, in the sense that many of the other languages which users are likely to use require the usage of macros to setup code in a way which allows the necessary state to be inserted for probabilistic programming functionality.

Jaynes sees all the code after macro expansion is completed, which allows Jaynes to introspect function call sites after state has been inserted by other libraries. This allows the possibility for Jaynes to construct special call sites to represent calls into other probabilistic programming libraries. These interfaces are a work in progress, but Jaynes should theoretically provide a _lingua franca_ for programs expressed in different probabilistic programming systems to communicate in a natural way, due to the nature of the context-oriented programming style facilitated by the system.

## Black-box extensions

```@docs
primitive
```

## Foreign models

```@docs
foreign
load_soss_fmi
load_gen_fmi
```
