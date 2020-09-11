module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes

function fn0(q::Float64, z::Float64, l)
    m = rand(:m, Normal(q, 1.0))
    q = rand(:q, Normal(z, 5.0))
    l = rand(:l, Bernoulli(l))
    m + q
end

fn1 = (q::Float64, z::Float64, m::Float64) -> begin
    t = rand(:z, Normal(5.0, 1.0))
    n = rand(:m, Normal(z, 1.0))
    l = rand(:l, fn0, q, m, m)
    if l > 5.0
        rand(:q, Normal(1.0, 5.0))
    else
        rand(:q, Normal(5.0, 3.0))
    end
    p = rand(:p, Normal(z, 3.0))
    t = rand(:t, Normal(p, 5.0))
    l + n
end

ret, cl = simulate(fn1, 10.0, 10.0, 0.5)
display(cl.trace)
println("Score: $(get_score(cl))")

# Simple diffs.
@time ret, cl, w, _ = update(cl, Δ(5.0, Change()),
                             Δ(5.0, Change()),
                             Δ(0.5, NoChange()))
display(cl.trace)
display(get_score(cl) - w)

end # module
