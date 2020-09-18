module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes

println("\n________________________\n")

@testset "Core" begin

    println("Compiler.")
    @time @testset "Compiler." begin
        include("compiler/update.jl")
    end

    println("Execution contexts.")
    @time @testset "Execution contexts." begin
        include("core/contexts.jl")
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

using Gen

@testset "Gen compatibility" begin
    println("Generative function interface.")
    @time @testset "Generative function interface." begin
        include("gen_compat/gen_fn_interface.jl")
    end
end

end #module
