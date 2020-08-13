module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

@time @testset "Execution contexts." begin
    include("contexts.jl")
end

@time @testset "Vectorized interfaces." begin
    include("vectorized.jl")
end

@time @testset "Importance sampling." begin
    include("importance_sampling.jl")
end

@time @testset "Particle filtering." begin
    include("particle_filter.jl")
end

@time @testset "Metropolis-Hastings." begin
    include("metropolis_hastings.jl")
end

@time @testset "Black-box extensions." begin
    include("blackbox.jl")
end

@time @testset "Selection query language." begin
    include("selection_query_language.jl")
end

@time @testset "Gradient learning." begin
    include("backpropagation.jl")
end

@time @testset "Smoke tests." begin
    include("bayeslinreg_smoke.jl")
end

end #module
