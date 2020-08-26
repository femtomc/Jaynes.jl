module GaussianMixtureModels

include("../src/Jaynes.jl")
using .Jaynes
using Distributions, StatsPlots, Random

# Set a random seed.
Random.seed!(3)

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
x = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)

# Visualization.
scatter(x[1,:], x[2,:], legend = false, title = "Synthetic Dataset")

@sugar GaussianMixtureModel(N) = begin

    # Draw the parameters for cluster 1.
    μ1 ~ Normal(0.0, 1.0)

    # Draw the parameters for cluster 2.
    μ2 ~ Normal(0.0, 1.0)

    μ = [μ1, μ2]

    # Uncomment the following lines to draw the weights for the K clusters 
    # from a Dirichlet distribution.

    #α = 1.0
    #w ~ Dirichlet(2, α)

    # Comment out this line if you instead want to draw the weights.
    w = [0.5, 0.5]

    # Draw assignments for each datum and generate it from a multivariate normal.
    k = [(:k => i) ~ Categorical(w) for i in 1 : N]
    x = [(:x => i) ~ MvNormal([μ[k[i]], μ[k[i]]], 1.) for i in 1 : N]
    return k
end

infer = (n_iters, n_samples) -> begin

    # Observations.
    obs = target([(:x => i, ) => [x[2*i], x[2*i + 1]] for i in 1 : 2 * N - 1])

    # MCMC targets.
    tg1 = target([(:μ1, ), (:μ2, )])
    tg2 = target([(:k => i, ) for i in 1 : N])

    # Generate an initial call site.
    ret, cl, _ = generate(obs, GaussianMixtureModel, N)

    # Run an MCMC chain.
    calls = []
    for i in 1 : n_iters
        @time begin
            cl, _ = mh(tg2, cl)
            cl, _ = hmc(tg1, cl)
            i % (n_iters / n_samples) == 0 && begin
                push!(calls, cl)
            end
        end
    end
    calls
end

n_iters = 3000
n_samples = 50
calls = infer(n_iters, n_samples)
μ1 = sum(map(calls) do cl
             cl[:μ1]
         end) / length(calls)

μ2 = sum(map(calls) do cl
             cl[:μ2]
         end) / length(calls)

println(μ1)
println(μ2)

end # module
