module Test

using Test

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

include("particle_filter.jl")

end
