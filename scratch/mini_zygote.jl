module MiniZygote

using IRTools
using IRTools: @dynamo, IR, var, argument!, xcall, recurse!

# Define the differentiation operator
J(::typeof(sin), x) = sin(x), ∂y -> ∂y * cos(x)
J(::typeof(cos), x) = cos(x), ∂y -> ∂y * -sin(x)
J(::typeof(*), a, b) = a*b, ∂c -> (b * ∂c, a * ∂c)

# Gradient operator uses the pullback generator J
∇(f, x...) = J(f, x...)[2](1)

# Dynamo!
@dynamo function J(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

# Something I should be able to differentiate...
function foo(x::Float64)
    y = sin(x)
    z = cos(y)
    return y * z
end

ir = @code_ir foo(5.0)
modified_ir = @code_ir J foo(5.0)

println(∇(sin, 0))
println(∇(cos, 0))

println(ir, "\n")
println(modified_ir, "\n")

end #module
