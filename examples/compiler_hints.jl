module CompilerHints

include("../src/Jaynes.jl")
using .Jaynes

model3 = @jaynes function baz(x::Int)
    y = 0
    for i in 1 : N
        y += {:y => i} ~ Normal(0.0, 1.0)
    end
end (hints)

end # module
