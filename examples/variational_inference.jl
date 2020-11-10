module VariationalInference

include("../src/Jaynes.jl")
using .Jaynes
using Gen

model = @jaynes (a₀::Float64, b₀::Float64, μ₀::Float64, λ₀::Float64, N::Int) -> begin
    τ ~ Gamma(a₀, b₀)
    μ ~ Normal(μ₀, 1 / (λ₀ * τ))
    x = [{:x => i} ~ Normal(μ, 1 / τ) for i in 1 : N]
    x
end

# Factorized approximation.
variational_family = @jaynes () -> begin
    α, β = learnable(:α), learnable(:β)
    τ ~ Gamma(α^2 + 0.01, β^2 + 0.01)
    μp, logσ = learnable(:μp), learnable(:logσ)
    μ ~ Normal(μp, exp(logσ))
end
init_param!(variational_family, [(:α, ) => 1.0, (:β, ) => 1.0,
                                 (:μp, ) => 0.0, (:logσ, ) => 0.0])

# Black box variational inference.
N = 100
observations = begin
    a₀ = 0.5
    b₀ = 1.0
    μ₀ = 0.0
    λ₀ = 3.0
    tr = simulate(model, (a₀, b₀, μ₀, λ₀, N))
    filter(k -> k[1] isa Pair, tr)
end
display(observations)

update = ParamUpdate(GradientDescent(1e-5, 100), 
                     variational_family => [(:α, ), (:β, ), (:μp, ), (:logσ, )])

@time elbo_estimate, traces, elbo_history = black_box_vi!(model, (0.5, 1.0, 0.0, 3.0, N), observations, variational_family, (), update; iters = 1000, verbose = true)
display(get_params(variational_family))

end # module
