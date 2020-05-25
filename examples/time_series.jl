module TimeSeries

include("../src/Walkman.jl")
using .Walkman
using Distributions

transition(z::Float64, addr) = rand(addr, Normal, (0.0, 1.0))
observation(z::Float64, addr) = rand(addr, Normal, (0.0, 1.0))

# TODO: re-write with Wavefolder.
function simulate(init_z_params::Tuple{Float64, Float64}, T::Int)
    μ, σ = init_z_params
    observations = Vector{Float64}(undef, T)
    z = rand(:z₀, Normal,(μ, σ))
    observations[1] = observation(z, :x₀)
    for i in 2:T
        z = transition(z, :z => i)
        observations[i] = observation(z, :x => i)
    end
    return observations
end

ctx, tr, weight = trace(simulate, ((0.5, 5.0), 50))
display(tr)

end # module
