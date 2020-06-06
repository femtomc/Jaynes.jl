module CassetteDSLs

using Cassette
using InteractiveUtils

Cassette.@context DSLCtx

@inline Cassette.overdub(ctx::DSLCtx, c::Function, args) = error("Call $c is not supported for this DSL.")

@inline Cassette.overdub(ctx::DSLCtx, c::Function) = error("Call $c is not supported for this DSL.")

ctx = DSLCtx()
fn = () -> for i in 1:10
    println(i)
end
low = @code_lowered fn()
println(low)

end # module
