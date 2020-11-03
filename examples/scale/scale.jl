module AutomaticAddressingScale

include("../../src/Jaynes.jl")
using .Jaynes
using Gen

# Define model of noisy weighing scale in Julia
function scale(N::Int)
    mass = rand(Normal(5, 1))
    obs = measure(mass, N)
end

function measure(mass::Real, N::Int)
    obs = Float64[]
    for i in 1 : N
        m = rand(Normal(mass, 2))
        push!(obs, m)
    end
    return obs
end

# Wrap in JFunction.
model = JFunction(scale, (Int, ), (false, ), false, Any)
tr = simulate(model, (10, ))
display(tr)

# Construct choicemap of noisy measurements
observations = constrain([(:measure, :rand_Normal => i) => 4 for i in 1:10])
tr, w = generate(model, (10, ), observations)
display(tr)

# Perform importance sampling to estimate the true mass
trs, ws, _ = importance_sampling(model, (10,), observations, 100)
mass_est = sum(tr[:rand_Normal] * exp(w) for (tr, w) in zip(trs, ws))
println(mass_est)

end # module
