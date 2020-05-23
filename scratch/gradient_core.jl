module GradientCore

using Cassette
using Cassette: recurse, Reflection
using IRTools
using MacroTools
using InteractiveUtils
using Core: CodeInfo

Cassette.@context GradientContext

Cassette.prehook(::GradientContext, f, args...) = println(f, args)

# This pass requires that the context has access to a map from param names to gradients.
function identity(ctx::Type{<:GradientContext}, ir::Reflection) where {S}
    println(ir)
    code_info = ir.code_info
    println(code_info)
    return code_info
end

const identity_pass = Cassette.@pass identity

function foo(y)
    x = 6 * y
    x += 10
    return x
end

ctx = GradientContext(pass = identity_pass, metadata = "Hi!")
println(Cassette.overdub(ctx, foo, 5))

end # module
