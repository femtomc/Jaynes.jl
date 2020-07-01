module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

@time @testset "Execution contexts." begin
    include("contexts.jl")
end

@time @testset "Particle filtering." begin
    include("particle_filter.jl")
end

@time @testset "Importance sampling." begin
    include("importance_sampling.jl")
end

@time @testset "Metropolis-Hastings." begin
    include("metropolis_hastings.jl")
end

@time @testset "Selection query language." begin
    include("selection_query_language.jl")
end

end #module
