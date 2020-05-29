module Simple

include("../src/Jaynes.jl")
using .Jaynes
using Distributions
using Cassette
using Cassette: disablehooks
using InteractiveUtils

function foo1(q::Float64)
    x = rand(:x, Normal, (q, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:foo2, foo2)
    return y + z
end

function foo2()
    z = rand(:z, Normal, (3.0, 1.0))
    return z
end

ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
low = @code_lowered Cassette.overdub(ctx, foo1, 5.0)
println("No pass:\n$(low)\n")

ctx = disablehooks(TraceCtx(pass = ignore_pass, metadata = UnconstrainedGenerateMeta(Trace())))
low = @code_lowered Cassette.overdub(ctx, foo1, 5.0)
println("Pass:\n$(low)\n")

ctx, tr, score = trace(ctx, foo1, (5.0, ))

end # module
