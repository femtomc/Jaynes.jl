# This example follows [this example from the Turing.jl tutorials.](https://turing.ml/dev/tutorials/1-gaussianmixturemodel/)

using Jaynes, Random
using GR
Jaynes.@load_chains();

# Set a random seed.

Random.seed!(3);

# Construct 30 data points for each cluster.

N = 30;

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.

μs = [-3.5, 0.0];

# Construct the data points.

x = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)
fig = GR.scatter(x[1,:], x[2,:], title = "Synthetic Dataset");
GR.savefig(joinpath(@OUTPUT, "gmm_synthetic_data.png"))

# \fig{output/gmm_synthetic_data}

# Here's the model! It's very easy to transfer models between Turing.jl and Jaynes. (Note that here, I'm not pre-allocating and then mutating arrays because the current AD engine is Zygote)

@sugar function GaussianMixtureModel(N)
    μ1 ~ Normal(0.0, 1.0)
    μ2 ~ Normal(0.0, 1.0)
    μ = [μ1, μ2]
    α = 1.0
    w ~ Dirichlet(2, α)
    k = [(:k => i) ~ Categorical(w) for i in 1 : N]
    x = [(:x => i) ~ MvNormal([μ[k[i]], μ[k[i]]], 1.) for i in 1 : N]
    return k
end;

# Here's a simple inference program with a custom kernel - the kernel proposes a Metropolis-Hastings move (from the prior) for the target addresses `:k => i` for `i in 1 : N`, followed by an HMC move for the continuous latent means, following by another MH move from the prior for the density coefficients drawn from the Dirichlet.

infer = (n_iters, n_samples) -> begin
    obs = target([(:x => i, ) => [x[2*i], x[2*i + 1]] for i in 1 : 2 * N - 1])
    tg1 = target([(:μ1, ), (:μ2, )])
    tg2 = target([(:k => i, ) for i in 1 : N])
    tg3 = target([(:w, )])
    ret, cl, _ = generate(obs, GaussianMixtureModel, N)
    calls = []
    for i in 1 : n_iters
        cl, _ = mh(tg2, cl)
        cl, _ = hmc(tg1, cl)
        cl, _ = mh(tg3, cl)
        i % Int(floor(n_iters / n_samples)) == 0 && begin
            push!(calls, cl)
        end
    end
    chn = chain(tg1, calls)
    chn
end;

n_iters = 10000
n_samples = 300
chn = infer(n_iters, n_samples)

# Now we can generate a nice plot of the chains using `StatsPlots`.

fig = StatsPlots.plot(chn);
StatsPlots.savefig(fig, joinpath(@OUTPUT, "gmm_chain_plot.png"))

# \fig{output/gmm_chain_plot}
