module BP

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

model = @sugar () -> begin
    x ~ Normal(0.0, 1.0)
    y ~ Normal(x, 3.0)
end

ir = @code_ir model()
display(ir)

end # module
