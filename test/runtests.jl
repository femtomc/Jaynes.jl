module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

include("particle_filter.jl")
include("importance_sampling.jl")
include("selection_query_language.jl")

end #module
