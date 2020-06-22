module TimeSeries

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

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

xs = [1, 1, 2, 2, 1]
init_obs = Jaynes.selection([(:x => 1, xs[1])])
tr = Jaynes.Trace()
ctx = Jaynes.TraceCtx(metadata = Jaynes.UnconstrainedGenerateMeta(tr), pass = Jaynes.ignore_pass)
ret = Jaynes.overdub(ctx, CategoricalHiddenMarkovModel, 10)
display(ctx.metadata.tr)
#trs, _, _ = Jaynes.importance_sampling(CategoricalHiddenMarkovModel, (1, ), init_obs, 50000)
#ps = Jaynes.initialize_filter(CategoricalHiddenMarkovModel, 
#                       (1, ),
#                       init_obs, 
#                       10000)    
#println(ps.lmle)
#for t=2:5
#    obs = Jaynes.selection([(:x => t, xs[t])])
#    filter_step!(ps, (t,), obs)
#    println(ps.lmle)
#end

end # module
