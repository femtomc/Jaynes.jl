# ------------ includes ------------ #

include("foreign_model_interfaces/blackbox.jl")
include("foreign_model_interfaces/soss.jl")
include("foreign_model_interfaces/turing.jl")
include("foreign_model_interfaces/gen.jl")

# ------------ Documentation -------------- #

function primitive end
function load_soss_fmi end
function load_turing_fmi end
function load_gen_fmi end

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
