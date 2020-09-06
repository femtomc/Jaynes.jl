module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes

println("\n________________________\n")

@testset "Core" begin
    println("Execution contexts.")
    @time @testset "Execution contexts." begin
        include("core/contexts.jl")
    end

    println("Vectorized interfaces.")
    @time @testset "Vectorized interfaces." begin
        include("core/vectorized.jl")
    end
    
    println("Gradients.")
    @time @testset "Gradients." begin
        include("core/gradients.jl")
    end

    println("Black-box extensions.")
    @time @testset "Black-box extensions." begin
        include("core/blackbox.jl")
    end

    println("Selection query language.")
    @time @testset "Selection query language." begin
        include("core/selection_query_language.jl")
    end

    println("Gradient learning.")
    @time @testset "Gradient learning." begin
    end
    println()
end

println("\n________________________\n")

@testset "Inference" begin
    println("Importance sampling.")
    @time @testset "Importance sampling." begin
        include("inference/importance_sampling.jl")
    end

    println("Particle filtering.")
    @time @testset "Particle filtering." begin
        include("inference/particle_filter.jl")
    end

    println("Metropolis-Hastings.")
    @time @testset "Metropolis-Hastings." begin
        include("inference/metropolis_hastings.jl")
    end
    
    println("Gradient-based inference.")
    @time @testset "Gradient-based inference." begin
        include("inference/gradient_based_inference.jl")
    end
    println()
end

println("\n________________________\n")

@testset "Smoke tests." begin
    println("Bayesian linear regression.")
    @time @testset "Bayesian linear regression." begin
        include("smoke_tests/bayeslinreg_smoke.jl")
    end
end

end #module
