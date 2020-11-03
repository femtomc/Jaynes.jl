module VariationalInference

include("../src/Jaynes.jl")
using .Jaynes
using Gen

model = @jaynes (a₀, μ₀, λ₀, N) -> begin
    b₀ = learnable(:b₀)
    τ ~ Gamma(a₀, b₀)
    μ ~ Normal(μ₀, 1 / (λ₀ * τ))
    x = [{:x => i} ~ Normal(μ, 1 / τ) for i in 1 : N]
    x
end
init_param!(model, (:b₀, ), 5.0)

display(model; show_all = true)

tr = simulate(model, (0.5, 3.0, 0.5, 50))
accumulate_param_gradients!(tr)
display(get_params_grads(model))

end # module
