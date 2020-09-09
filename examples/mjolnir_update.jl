module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

Distributions.Normal(d::Diffed, x) = Distributions.Normal(d.value, x)

fn0 = (q, z) -> begin
    m = rand(:m, Normal(q, 1.0))
    q = rand(:q, Normal(z, 5.0))
    m + q
end

fn1 = (q, z, m) -> begin
    n = rand(:m, Normal(z, 1.0))
    l = rand(:l, fn0, q, m)
    l
end

ret, cl = simulate(fn1, 10.0, 10.0, 1.0)
display(cl.trace)
println("Score: $(get_score(cl))")
ret, cl, w, _ = update(cl, Δ(10.0, NoChange()), Δ(5.0, ScalarDiff(-5.0)), Δ(1.0, NoChange()))
display(cl.trace)
println("Score: $(get_score(cl))")
println("Weight: $w")

end # module
