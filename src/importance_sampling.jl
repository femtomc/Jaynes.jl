function importance_sampling(model::Function, 
                             args::Tuple,
                             observations::Dict{Address, Real},
                             num_samples::Int)
    trs = [Trace(observations) for i in 1:num_samples]
    res = map(trs) do tr
        tr() do
            model(args...)
        end
        (tr, tr.score)
    end
    lws = map(res) do (_, s)
        s
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             num_samples::Int)
    trs = [Trace() for i in 1:num_samples]
    lws = Vector{Float64}(undef, num_samples)
    i = 1
    res = map(trs) do tr
        tr() do
            model(args...)
        end
        lws[i] = tr.score
        i += 1
        tr
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end

function importance_sampling(model::Function, 
                             args::Tuple,
                             proposal::Function,
                             proposal_args::Tuple,
                             obs::Dict{Address, Real},
                             num_samples::Int)
    trs = [Trace() for i in 1:num_samples]
    lws = Vector{Float64}(undef, num_samples)
    i = 1
    res = map(trs) do prop
        # Propose.
        prop() do
            proposal(proposal_args...)
        end

        # Merge proposals and observations.
        constraints = merge(obs, prop.chm)
        tr = Trace(constraints)

        # Generate.
        tr() do
            model(args...)
        end

        # Track score.
        lws[i] = tr.score - prop.score
        i += 1
        tr
    end
    ltw = lse(lws)
    lmle = ltw - log(num_samples)
    lnw = lws .- ltw
    return trs, lnw, lmle
end
