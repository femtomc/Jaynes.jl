module Effects

include("../src/Jaynes.jl")
using .Jaynes
using Distributions
using IRTools

function foo()
    z = rand(:z, Normal, 0.1, 1.0)
    for i in 1:10
        z = rand(:z => i, Normal, z, 1.0)
    end
    return z
end

ir = @code_ir foo()
println(ir)

end #module
