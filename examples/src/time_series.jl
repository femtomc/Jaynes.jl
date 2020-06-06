module TimeSeries

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions
using Profile
using PProf

transition(z::Float64, addr) = rand(addr, Normal, (0.0, 1.0))
observation(z::Float64, addr) = rand(addr, Normal, (0.0, 1.0))

# TODO: re-write with Wavefolder.
function simulate(init_z_params::Tuple{Float64, Float64}, T::Int)
    μ, σ = init_z_params
    observations = Vector{Float64}(undef, T)
    z = rand(:z₀, Normal,(μ, σ))
    observations[1] = observation(z, :x₀)
    for i in 2:T
        z = rand(:trans, transition, (z, :z => i))
        observations[i] = rand(:obs, observation, (z, :x => i))
    end
    return observations
end

sim = () -> begin
    ctx = Generate(Trace())
    for i in 1:1000
        ctx, tr, weight = trace(ctx, simulate, ((0.5, 5.0), 500))
        reset_keep_constraints!(ctx)
    end
end

sim()
Profile.clear_malloc_data()
@pprof sim()

end # module
