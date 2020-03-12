module InfCompDynamo

using Jaynes

using Flux
using Flux: Recur, RNNCell
using CuArrays

using IRTools
using IRTools: blocks
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, Statement, isexpr

using Distributions

inf_comp = InferenceCompiler(Recur(RNNCell(10, 10)))

function stoked_fantastic_bruh()
    x = rand(Normal(:x, 0.0, 1.0))
    y = rand(Normal(:y, x, 1.0))
    return y
end

ir = @code_ir stoked_fantastic_bruh()
println("Original IR:\n$(ir)")

transformed = @code_ir inf_comp stoked_fantastic_bruh()
println("\nTransformed:\n$(transformed)")

inf_comp() do
    stoked_fantastic_bruh()
end

end # module
