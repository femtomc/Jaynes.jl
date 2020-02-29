using IRTools: IR, @dynamo, recurse!

mul(a, b) = a * b

@dynamo function roundtrip(a...)
    ir = IR(a...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

roundtrip(::typeof(mul), a, b) = a + b

println(@code_ir mul(4, 3))
ir = @code_ir roundtrip mul(4, 3)
println(ir)

roundtrip() do
    mul(4, 3)
end
