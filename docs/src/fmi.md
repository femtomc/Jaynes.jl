## Black-box extensions

Jaynes is equipped with the ability to extend the tracer to arbitrary black-box code, as long as the user can provide a `logpdf` for the call

```julia
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end


cl = Jaynes.call(Trace(), rand, :geo, geo, 0.3)
display(cl.trace; show_values = true)
```

will produce

```
  __________________________________

               Addresses

 geo : 4
  __________________________________
```

## Foreign models

Due to the design and implementation as an IR metaprogramming tool, Jaynes sits at a slightly privileged place in the probabilistic programming ecosystem, in the sense that many of the other languages which users are likely to use require the usage of macros to setup code in a way which allows the necessary state to be inserted for probabilistic programming functionality.

Jaynes sees all the code after macro expansion is completed, which allows Jaynes to introspect function call sites after state has been inserted by other libraries. This allows the possibility for Jaynes to construct special call sites to represent calls into other probabilistic programming libraries. These interfaces are a work in progress, but Jaynes should theoretically provide a _lingua franca_ for programs expressed in different probabilistic programming systems to communicate in a natural way, due to the nature of the context-oriented programming style facilitated by the system.
