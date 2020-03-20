module LoweredPass

using MacroTools
using MacroTools: postwalk

using JSON

using IRTools
using Cassette
using InteractiveUtils: @code_lowered

insert_addr_in_call = (expr, addr) -> postwalk(x -> @capture(x, f_(xs__)) ? (f isa GlobalRef && f.name == :rand ? :($f($addr, $(xs...))) : x) : x, expr)

function transform_sample_statements(ci::Core.CodeInfo)
    lowered = copy(ci)
    code = lowered.code

    # Mapping the code...
    identifier_dict = Dict(map(x -> (Core.SlotNumber(x[1]) => x[2]), enumerate(lowered.slotnames)))
    SSA_dict = Dict(map(x -> (Core.SSAValue(x[1]) => x[2]), enumerate(code)))

    # Insertion.
    transformed = map(line -> (line isa Expr && line.args[1] isa Core.SlotNumber) ? insert_addr_in_call(line, identifier_dict[line.args[1]]) : line, code)

    # Dependency insertion.
    #transformed = map(expr -> postwalk(y -> (y isa Core.SSAValue && @capture(SSA_dict[y], f_(xs__)) && (f isa GlobalRef && f.name == :rand)) ? insert_addr_in_call(transformed[y.id], y) : y, expr), transformed)

    # Insert new code.
    lowered.code = transformed
    return lowered
end

function foo(x::Float64)
    l = 0
    y = rand(Normal(x, 1.0))
    q = rand(Normal(y, 5.0))
    z = x + rand(Normal(y, 5.0))
    while rand(Normal(10, 10)) < 20
        z += rand(Normal(z, 0.5))
    end
    z = z + bar(z)
    return z
end

lowered = @code_lowered foo(5.0)
ir = @code_ir foo(5.0)
transformed = transform_sample_statements(lowered)
println(transformed)

end # module
