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

function CategoricalHiddenMarkovModel(time_period::Int)
    # Initial.
    init_z = rand(:z => 1, Categorical([0.5, 0.5]))
    latents = Vector{Int}(undef, time_period)
    latents[1] = init_z
    observations = Vector{Int}(undef, time_period)

    # Observation model.
    obs = (z, i) -> begin
        if z == 1
            rand(:x => i, Categorical([0.5, 0.5]))
        else
            rand(:x => i, Categorical([0.2, 0.8]))
        end
    end

    # Transition model.
    trans = (prev_z, i) -> begin
        if prev_z == 1
            rand(:z => i, Categorical([0.5, 0.5]))
        else
            rand(:z => i, Categorical([0.2, 0.8]))
        end
    end

    observations[1] = obs(init_z, 1)
    z = init_z

    for i in 2:time_period
        z = trans(z, i)
        observation = obs(z, i)
        latents[i] = z
        observations[i] = observation
    end
    return observations
end

function LinearGaussianStateSpaceModel
end


@testset "Particle filtering" begin

    @testset "Categorical hidden Markov model" begin
        tol = 0.1
        checks = [-1.05, 
                  -2.18, 
                  -2.57, 
                  -2.907, 
                  -4.19]
        xs = [1, 1, 2, 2, 1]
        lmles = []

        # Testing.
        init_obs = Jaynes.selection([(:x => 1, xs[1])])
        ps = Jaynes.initialize_filter(CategoricalHiddenMarkovModel, 
                                      (1, ),
        init_obs, 
        50000)    
        push!(lmles, ps.lmle)
        for t=2:5
            obs = Jaynes.selection([(:x => t, xs[t])])
            Jaynes.filter_step!(ps, (t,), obs)
            push!(lmles, ps.lmle)
        end
        map(enumerate(checks)) do (k, v)
            @test v ≈ lmles[k] atol = tol
        end
    end
end
