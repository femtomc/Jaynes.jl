module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

fn0 = q -> begin
    m = rand(:m, Normal(q, 1.0))
    m
end

ret, cl = simulate(fn0, 5.0)
ret, cl, w, _ = update(cl, Δ(10.0, ScalarDiff(5.0)))
display(cl.trace)

end # module
