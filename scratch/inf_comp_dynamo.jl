using Jaynes

using IRTools
using IRTools: blocks
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, Statement, isexpr

inf_comp = InferenceCompiler(GPU(Recur(RNNCell(10, 10))))

@dynamo function (inf_comp::InferenceCompiler)(m...)
    ir = IR(m...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

function (inf_comp::InferenceCompiler)(call_norm::typeof(Normal), a::Float64, b::Float64)
    name = gensym()
    params = send!(GPU(Request(name, [0.0, 0.0])), inf_comp)
    println(params)
    return call_norm(name, params[1], exp(params[2]))
end

function (inf_comp::InferenceCompiler)(::typeof(rand), a::T) where {T <: Randomness}
    result = rand(a)
    send!(GPU(Post(a.name, [result])), inf_comp)
    result
end

function stoked_fantastic_bruh()
    x = rand(Normal(0.0, 1.0))
    y = rand(Normal(x, 1.0))
    return y
end

ir = @code_ir stoked_fantastic_bruh()
println(ir)

transformed = @code_ir inf_comp stoked_fantastic_bruh()
println(transformed)

inf_comp() do
    stoked_fantastic_bruh()
end
