module PaperExample

include("../src/Jaynes.jl")
using .Jaynes
using IRTools

expensive_deterministic_call(p) = (sleep(3); 5.0)

model = @sugar (p1::Float64, p2::Float64) -> begin
    z ~ Normal(0.0, 1.0)
    q1 ~ Normal(p1, exp(z))
    q2 ~ Normal(p2, exp(z))
    x ~ Normal(q1 + q2, 1.0)
    x
end

println(@code_ir(model(3.0, 5.0)))
ret, cl = simulate(model, 3.0, 5.0)
display(cl.trace)

obs = static([(:q1, ) => 5.0])
ret, cl, _ = update(obs, cl, Δ(3.0, NoChange()), Δ(6.0, ScalarDiff(1.0)))
@time ret, cl, _ = update(obs, cl, Δ(3.0, NoChange()), Δ(6.0, ScalarDiff(1.0)))
display(cl.trace)

end # module
