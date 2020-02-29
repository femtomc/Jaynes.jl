module TestInferenceCompilation

include("../src/Jaynes.jl")
using .Jaynes

inf_comp = InferenceCompiler(GPU(Recur(RNNCell(10, 10))))
println(inf_comp)

end
