module Subtypes

using Distributions

dists = map(x -> Symbol(x), subtypes(Distribution))
println(dists)

end
