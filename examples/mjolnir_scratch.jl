module MjolnirScratch

include("../src/Jaynes.jl")
using .Jaynes

fn0 = q -> begin
    x = 5.0 + q
    z = 10.0 + x
    q = z * x + 5.0
    m = rand(:m, Normal(0.0, 1.0))
    x
end

fn1 = q -> begin
    x = 5.0 + q
    z = 10.0 + x
    q = z * x + 5.0
    m = rand(:m, Normal(0.0, 1.0))
    x
end

fn2 = (q, z, m) -> begin
    l = rand(:z, fn0, z)
    l = rand(:m, fn0, q)
    l = rand(:n, fn1, m)
    l * 2
end

tr = _pushforward(fn2, Δ(5.0, NoChange()), Δ(10.0, UndefinedChange()), Δ(10.0, NoChange()))
display(tr)

end # module
