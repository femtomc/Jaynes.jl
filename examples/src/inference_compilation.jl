module Profiling

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

function foo1()
    x = rand(:x, Normal, (3.0, 10.0))
    y = rand(:y, Normal, (x + 15.0, 0.3))
    return y
end

ctx = inference_compilation(foo1, (), :y; batch_size = 256, epochs = 100)
obs = constraints([(:y, 10.0)])
fn = (ctx, obs) -> while true
    ctx, tr, score = trace(ctx, obs)
    display(tr)
    sleep(3)
end
fn(ctx, obs)

end # module
