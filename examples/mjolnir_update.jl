module MjolnirUpdate

include("../src/Jaynes.jl")
using .Jaynes

function expensive_call(q)
    sleep(3)
    l = [1.0 for i in 1 : 100000]
    m = rand(:m, Normal(q, 3.0))
    return m
end

function fn0(q::Float64, z::Float64)
    m = rand(:m, Normal(q, 1.0))
    q = rand(:q, Normal(z, 5.0))
    l = rand(:l, Normal(m + q, 3.0))
    n = rand(:n, Bernoulli(0.3))
    p = rand(:t, expensive_call, q)
    q + l + p
end

ret, cl = simulate(fn0, 10.0, 10.0)
display(cl.trace)
update(cl, Δ(10.0, NoChange()), Δ(10.0, NoChange()))
@time update(cl, Δ(10.0, NoChange()), Δ(10.0, NoChange()))
display(cl.trace)

end # module
