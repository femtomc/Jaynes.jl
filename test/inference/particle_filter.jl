function CategoricalMarkovChain(time_period::Int)
    # Initial.
    init_x = rand(:x => 1, Categorical([0.5, 0.5]))
    observations = Vector{Int}(undef, time_period)
    observations[1] = init_x
    x = init_x
    for i in 2:time_period
        if x == 1
            trans = rand(:x => i, Categorical([0.5, 0.5]))
        else
            trans = rand(:x => i, Categorical([0.8, 0.2]))
        end
        observations[i] = trans
    end
    return observations
end

# ------------ CHMM ------------ #

# Observation model.
obs = z -> begin
    if z == 1
        rand(:x, Categorical([0.5, 0.5]))
    else
        rand(:x, Categorical([0.2, 0.8]))
    end
end

# Transition model.
trans = prev_z -> begin
    if prev_z == 1
        rand(:z, Categorical([0.5, 0.5]))
    else
        rand(:z, Categorical([0.2, 0.8]))
    end
end

function CategoricalHiddenMarkovModel(time_period::Int)
    # Initial.
    init_z = rand(:z => 1, Categorical([0.5, 0.5]))
    latents = Vector{Int}(undef, time_period)
    latents[1] = init_z
    observations = Vector{Int}(undef, time_period)

    observations[1] = rand(:obs => 1, obs, init_z)
    z = init_z

    for i in 2:time_period
        z = rand(:trans => i, trans, z)
        observation = rand(:obs => i, obs, z)
        latents[i] = z
        observations[i] = observation
    end
    return observations
end


@testset "Particle filtering" begin

    @testset "Categorical hidden Markov model" begin
        tol = 0.1
        checks = [-1.05, -2.18, -2.57, -2.907, -4.19]
        xs = [1, 1, 2, 2, 1]
        lmles = []

        # Testing.
        init_obs = Jaynes.target([(:obs => 1, :x) => xs[1]])
        ps = Jaynes.initialize_filter(init_obs, 50000, CategoricalHiddenMarkovModel, (1, ))
        push!(lmles, get_lmle(ps))
        for t=2:5
            obs = Jaynes.target([(:obs => t, :x) =>  xs[t]])
            Jaynes.filter_step!(obs, ps, (Δ(t, IntDiff(1)), ))
            push!(lmles, get_lmle(ps))
        end
        map(enumerate(checks)) do (k, v)
            @test v ≈ lmles[k] atol = tol
        end
    end
end
