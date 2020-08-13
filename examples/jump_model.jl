module JumpModel

include("../src/Jaynes.jl")
using .Jaynes

function generative_model_jaynes_hmc(n_data)
    μ_0    = 0.0
    λ      = 1
    σ_μ    = 1.0
    σ_data = 1.0 
    ρ      = 0.25 # rand(:jump_prob, Beta(2,5)) # Jump probability
    μ      = μ_0     
    for (k, n) in enumerate(n_data)
        jump = rand(:jump => k, Bernoulli(ρ))
        s = (k == 1 || jump) ? σ_μ : 1e-4
        μ = rand(:mu => k, Normal(μ, s))
        for j in 1:n
            rand(:data => k => j, Normal(μ, σ_data))
        end
    end
end

infer = () -> begin
    m = 5
    N = [m for i in 1 : 8]
    ret, cl = simulate(generative_model_jaynes_hmc, N)
    sel = get_selection(cl)
    data = filter(k -> k[1] == :data, y -> true, sel)
    sel1 = selection(map(dump_queries(filter(k -> k[1] == :mu, y-> true, sel))) do k
                        k[1]
                    end)
    sel2 = selection(map(dump_queries(filter(k -> k[1] == :jump, y-> true, sel))) do k
                        k[1]
                    end)
    
    gt = [mean([data[:data => i => j] for j in 1 : 5]) for i in 1 : 8]
    println("(Ground truth):")
    map(gt) do g
        println(g)
    end
    
    obs = filter(k -> k[1] == :data, y -> true, get_selection(cl))
    n_samples = 5000
    @time ps, lnw = importance_sampling(obs, n_samples, generative_model_jaynes_hmc, (N, ))
    display(get_selection(ps.calls[1]))

    # IS.
    ps, lnw = is(data, 50000, generative_model_jaynes_hmc, (N, ))
    zipped = zip(ps.calls, lnw)
    est_μ = sum(map(zipped) do (cl, w)
                    [get_ret(cl[:mu => i]) for i in 1 : 8] * exp(w)
                end)
    println("\n(IS):")
    map(est_μ) do μ
        println(μ)
    end

    # HMC.
    calls = []
    for i in 1 : 5000
        cl, _ = hmc(sel1, cl)
        cl, _ = mh(sel2, cl)
        cl, _ = mh(sel1, cl)
        i > 500 && i % 15 == 0 && push!(calls, cl)
    end
    est_μ = sum(map(calls) do cl
                    [get_ret(cl[:mu => i]) for i in 1 : 8]
                end) / length(calls)
    println("\n(Custom kernel):")
    map(est_μ) do μ
        println(μ)
    end

end

infer()

end # module
