module Simple

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions
using Cassette
using Cassette: disablehooks
using InteractiveUtils

function foo1(q::Float64)
    x = rand(:x, Normal, (q, 1.0))
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:foo2, foo2)
    r = y + 20
    return y + z + r
end

function foo2()
    z = rand(:z, Normal, (3.0, 1.0))
    return z
end

function mul_switch!(ir)
    ir = MacroTools.prewalk(ir) do x
        x isa GlobalRef && x.name == :(+) && return GlobalRef(Base, :*)
        x
    end
    return ir
end


ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
#low = @code_lowered Cassette.overdub(ctx, foo1, 5.0)
#println("No pass:\n$(low)\n")
ret = Cassette.overdub(ctx, foo1, 5.0)
println(ret)

compose_pass = passfold(TraceCtx, mul_switch!)

ctx = disablehooks(TraceCtx(pass = compose_pass, metadata = UnconstrainedGenerateMeta(Trace())))
#low = @code_lowered Cassette.overdub(ctx, foo1, 5.0)
#println("Pass:\n$(low)\n")
ret = Cassette.overdub(ctx, foo1, 5.0)
println(ret)

end # module
