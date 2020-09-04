module GaussianMixtureModels

using Random
include("../src/Jaynes.jl")
using .Jaynes

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
    μ1 ~ Normal(-1.0, 3.0)

    # Draw the parameters for cluster 2.
    μ2 ~ Normal(0.0, 3.0)

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

dnn1 = Dense(120, 4)
dnn2 = Chain(Dense(120, 2), softmax)

var = @sugar (cl, data) -> begin
    (param1, param2, param3, param4) <- dnn1(data)
    μ1 ~ Normal(param1, exp(param3))
    μ2 ~ Normal(param2, exp(param4))
    w <- dnn2(data)
    k = [(:k => i) ~ Categorical(w) for i in 1 : 2 * N]
end

# Observations.
obs = target([(:x => i, ) => [x[2*i], x[2*i + 1]] for i in 1 : 2 * N - 1])
display(obs)
data = collect(Iterators.flatten(x))

# Train the variational network.
train = K -> begin
    opt = ADAM(5e-4, (0.9, 0.9))
    elbos = Float64[]
    for i in 1 : K
        elbo, cl = nvi!(opt,
                        obs, 
                        var, (nothing, data),
                        GaussianMixtureModel, (N, );
                        gs_samples = 100)
        push!(elbos, elbo)
        println(elbo)
        display(lineplot(elbos))
    end
end
train(100)

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
        #cl, acc = mh(tg2, cl, var, (data, ))
        cl, acc = hmc(tg1, cl)
        acc && display((cl[:μ1], cl[:μ2]))
        i % (n_iters / n_samples) == 0 && begin
            push!(calls, cl)
        end
    end
    calls
end

println("Neural MCMC.")
n_iters = 10000
n_samples = 500
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
