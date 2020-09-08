module MjolnirScratch

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

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

fn2 = (q, z, m, b) -> begin
    l = rand(:z, fn0, z)
    q = rand(:m, fn0, q)
    l = rand(:n, fn1, q)
    m = 0
    if b
        m = rand(:q, fn2, l) + m
    end
    v = l * q * m
    if v > 10.0
        return v
    else
        return l
    end
end

tr = _pushforward(fn2, Δ(5.0, NoChange()), 
                       Δ(10.0, ScalarDiff(5.0)), 
                       Δ(10.0, ScalarDiff(5.0)), 
                       Δ(true, BoolDiff(true)))
display(tr)

end # module
