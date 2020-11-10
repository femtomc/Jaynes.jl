module CombinatorsSpecialization

include("../src/Jaynes.jl")
using .Jaynes
using Gen

ker = @jaynes (i::Int64, x::Float64) -> begin
    x += x ~ Normal(x, 3.0)
    x
end
uc = Unfold(ker)

model = @jaynes (N::Int, p1::Float64, p2::Float64) -> begin
    z ~ Normal(0.0, 1.0)
    x ~ ker(N, z)
    x
end

tr = simulate(model, (10, 3.0, 5.0))

obs = constrain([(:x, :x, 1) => 5.0])
display(obs)
tr, _ = update(tr, (10, 3.0, 6.0), (NoChange(), NoChange(), ScalarDiff(1.0)), obs)
@time tr, _ = update(tr, (10, 3.0, 6.0), (NoChange(), NoChange(), ScalarDiff(1.0)), obs)

end # module
