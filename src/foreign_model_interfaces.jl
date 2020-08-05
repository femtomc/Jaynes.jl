# ------------ includes ------------ #

include("foreign_model_interfaces/blackbox.jl")
include("foreign_model_interfaces/soss.jl")
include("foreign_model_interfaces/turing.jl")
include("foreign_model_interfaces/gen.jl")
include("foreign_model_interfaces/flux.jl")

function primitive end
function load_soss_fmi end
function load_turing_fmi end
function load_flux_fmi end
function load_gen_fmi end

foreign(addr::A, args...) where A <: Address = error("(foreign) call with address $addr evaluated outside of the tracer.\nThis normally occurs because you're not matching the dispatch correctly.")

# ------------ Documentation -------------- #

@doc(
"""
```julia
@primitive function logpdf(fn::typeof(foo), args, foo_ret)
    ...
end
```

`@primitive` is a convenience metaprogramming construct which derives contextual dispatch definitions for functions which you'd like the tracer to "summarize" to a single site. This is one mechanism which gives the user more control over the structure of the choice map structure in their programs.

Example:

```julia
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 1 : 1 + rand(:geo, geo, p)

# Define as primitive.
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end

function foo()
    ret = rand(:geo, geo, 0.3)
    ret
end
```

The above program would summarize the trace into a single choice site for `:geo`, as if `geo` was a primitive distribution.

```julia
  __________________________________

               Addresses

 geo : 42
  __________________________________
```
""", primitive)

@doc(
"""
```julia
foreign(addr::A, m, args...) where A <: Address
```

Activate a foreign model interface. The tracer will treat this is a specialized call site, depending on the type of `m`. Currently supports `typeof(m) <: Soss.Model` and `typeof(m) <: Gen.GenerativeFunction`.

""", foreign)

@doc(
"""
```julia
Jaynes.@load_soss_fmi
```

`@load_soss_fmi` loads the [Soss.jl](https://github.com/cscherrer/Soss.jl) foreign model interface extension. This allows you to utilize `Soss` models in your modeling.

Example:

```julia
Jaynes.@load_soss_fmi()

# A Soss model.
m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(5)
end

bar = () -> begin
    x = rand(:x, Normal(5.0, 1.0))
    soss_ret = foreign(:foo, m, (σ = x,))
    return soss_ret
end
```

This interface currently supports all the inference interfaces (e.g. `simulate`, `generate`, `score`, `regenerate`, `update`, `propose`) which means that you can use any of the inference algorithms in the standard inference library.
""", load_soss_fmi)

@doc(
"""
```julia
Jaynes.@load_gen_fmi
```

`@load_gen_fmi` loads the [Gen.jl](https://www.gen.dev/) foreign model interface extension. This allows you to utilize `Gen` models (in any of Gen's DSLs) in your modeling.

Example:

```julia
Jaynes.@load_gen_fmi()

@gen (static) function foo(z::Float64)
    x = @trace(normal(z, 1.0), :x)
    y = @trace(normal(x, 1.0), :y)
    return x
end

Gen.load_generated_functions()

bar = () -> begin
    x = rand(:x, Normal(0.0, 1.0))
    return foreign(:foo, foo, x)
end

ret, cl = Jaynes.simulate(bar)
```

This interface currently supports all the inference interfaces (e.g. `simulate`, `generate`, `score`, `regenerate`, `update`, `propose`) which means that you can use any of the inference algorithms in the standard inference library.
""", load_gen_fmi)
