module GaussianMixtureModels

using Random
include("../src/Jaynes.jl")
using .Jaynes
Jaynes.@load_flux_fmi()

# Set a random seed.
Random.seed!(3)

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
x = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)

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
    k = [(:k => i) ~ Categorical(w) for i in 1 : 2 * N]
    x = [(:x => i) ~ MvNormal([μ[k[i]], μ[k[i]]], 1.) for i in 1 : 2 * N - 1]
    return k
end

dnn1 = Chain(Dense(120, 60), Dense(60, 2))
dnn2 = Chain(Dense(120, 30), Dense(30, 2), softmax)

var = @sugar (cl, data) -> begin
    params <- dnn1(data)
    μ1 ~ Normal(params[1], 1.0)
    μ2 ~ Normal(params[2], 1.0)
    w <- dnn2(data)
    k = [(:k => i) ~ Categorical(w) for i in 1 : 2 * N]
end

# Observations.
obs = target([(:x => i, ) => [x[2*i], x[2*i + 1]] for i in 1 : 2 * N - 1])
display(obs)
data = collect(Iterators.flatten(x))


# Train the variational network.
elbows, cls = nvi!(obs, 
                   var, (nothing, data),
                   GaussianMixtureModel, (N, );
                   opt = ADAM(2e-3, (0.9, 0.999)), 
                   n_iters = 500, 
                   gs_samples = 300)

#params, elbows, cls = advi(obs, 
#                           params,
#                           var, (nothing, data),
#                           GaussianMixtureModel, (N, );
#                           opt = ADAM(2e-3, (0.9, 0.999)), 
#                           n_iters = 200, 
#                           gs_samples = 300)

# Neural MCMC.
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
            cl, _ = mh(tg2, cl, var, (data, ))
            i % (n_iters / n_samples) == 0 && begin
                push!(calls, cl)
            end
        end
    end
    calls
end

println("Neural MCMC.")
n_iters = 10000
n_samples = 700
calls = infer(n_iters, n_samples)
μ1 = sum(map(calls) do cl
             cl[:μ1]
         end) / length(calls)

μ2 = sum(map(calls) do cl
             cl[:μ2]
         end) / length(calls)

println(μ1)
println(μ2)

# Neural IS.
println("Neural IS.")
ps, lnw = is(obs, 7000, GaussianMixtureModel, (N, ), var, (nothing, data))
μ1 = sum(map(zip(ps.calls, lnw)) do (cl, w)
             cl[:μ1] * exp(w)
         end)
μ2 = sum(map(zip(ps.calls, lnw)) do (cl, w)
             cl[:μ2] * exp(w)
         end)

println(μ1)
println(μ2)

# IS
println("IS.")
ps, lnw = is(obs, 10000, GaussianMixtureModel, (N, ))
μ1 = sum(map(zip(ps.calls, lnw)) do (cl, w)
             cl[:μ1] * exp(w)
         end)
μ2 = sum(map(zip(ps.calls, lnw)) do (cl, w)
             cl[:μ2] * exp(w)
         end)

println(μ1)
println(μ2)

end # module
