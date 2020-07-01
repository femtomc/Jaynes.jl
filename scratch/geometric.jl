module Geometric

include("../src/Jaynes2.jl")
using .Jaynes2
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)
tr = Jaynes2.Trace()
call = tr(geo, 0.5)
display(tr; show_values = true)
@time calls, lnw, lmle = Jaynes2.importance_sampling(geo, (0.05, ))

end # module
