module VariationalInference

include("../src/Jaynes.jl")
using .Jaynes
using Gen

model = @jaynes (a₀, b₀, μ₀, λ₀, N) -> begin
    τ ~ Gamma(a₀, b₀)
    μ ~ Normal(μ₀, 1 / (λ₀τ))
    x = [{:x => i} ~ Normal(μ, 1 / τ) for i in 1 : N]
    x
end

# Factorized approximation.
variational_family = @jaynes () -> begin
    α, β = learnable(:α), learnable(:β)
    τ ~ Gamma(α, β)
    μp, logσ = learnable(:μp), learnable(:logσ)
    μ ~ Normal(μp, exp(logσ))
end

init_param!(variational_family, [(:α, ) => 1.0, (:β, ) => 1.0,
                                 (:μp, ) => 0.0, (:logσ, ) => 0.0])
# Black box variational inference.
N = 100
observations = target([(:x => i, ) => rand(Normal(3.0, 0.3)) for i in 1 : N])
update = ParamUpdate(GradientDescent(0.001, 100), 
                     variational_family => [(:α, ), (:β, ), (:μp, ), (:logσ, )])

#elbo_estimate, traces, elbo_history = black_box_vi!(model, (0.0, 0.0, 0.0, 0.0),
#                                                    observations,
#                                                    variational_family, (),
#                                                    update)

end # module
