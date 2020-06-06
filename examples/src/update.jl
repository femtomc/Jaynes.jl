module TestUpdate

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

function bar()
    val = rand(:a, Bernoulli, (0.3,)) == 1
    if rand(:b, Bernoulli, (0.4,)) == 1
        val = rand(:c, Bernoulli, (0.6, )) == 1 && val
    else
        val = rand(:d, Bernoulli, (0.1,)) == 1 && val
    end
    val = rand(:e, Bernoulli, (0.7,)) == 1 && val
    return val
end

ctx, tr, score = trace(bar)
display(tr)
update_ctx = Update(tr, constraints([(:b, 0), (:d, 1)]))
ctx, tr, score, dis = trace(update_ctx, bar)
display(tr)
println(dis)

end # module
