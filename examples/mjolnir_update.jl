module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

Distributions.Normal(d::Diffed, x) = Distributions.Normal(d.value, x)

fn0 = (q, z) -> begin
    m = rand(:m, Normal(q, 1.0))
    d = Normal(z, 5.0)
    println(d)
    q = rand(:q, d)
    m + q
end

ret, cl = simulate(fn0, 10.0, 10.0)
display(cl.trace)
println(get_score(cl))
ret, cl, w, _ = update(cl, Δ(10.0, NoChange()), Δ(5.0, ScalarDiff(-5.0)))
display(cl.trace)
println(get_score(cl))
println(w)

end # module
