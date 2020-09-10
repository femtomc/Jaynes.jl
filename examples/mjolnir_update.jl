module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

Distributions.Normal(v::D, x) where D <: Diffed = Normal(v.value, x)

fn0 = (q::Float64, z::Float64) -> begin
    m = rand(:m, Normal(q, 1.0))
    q = rand(:q, Normal(z, 5.0))
    m + q
end

fn1 = (q::Float64, z::Float64, m::Float64) -> begin
    t = rand(:z, Normal(5.0, 1.0))
    n = rand(:m, Normal(z, 1.0))
    l = rand(:l, fn0, q, m)
    if l > 5.0
        rand(:q, Normal(1.0, 5.0))
    else
        rand(:q, Normal(5.0, 3.0))
    end
    p = rand(:p, Normal(z, 3.0))
    t = rand(:t, Normal(p, 5.0))
    l + n
end

ret, cl = simulate(fn1, 10.0, 10.0, 1.0)
display(cl.trace)
println("Score: $(get_score(cl))")

# Simple diffs.
ret, cl, w, _ = update(cl, Δ(5.0, ScalarDiff(-5.0)), 
                           Δ(5.0, ScalarDiff(-5.0)), 
                           Δ(1.0, NoChange()))
display(cl.trace)
display(get_score(cl) - w)

end # module
