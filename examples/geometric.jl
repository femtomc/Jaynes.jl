module Geo

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(Bernoulli(p)) ? 1 : 1 + geo(p)

# Define as primitive.
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end

# Example trace.
tg = target([(:geo, ) => 45])
ret, cl, _ = generate(tg, () -> rand(:geo, geo, 0.05))
display(cl.trace)

end # module
