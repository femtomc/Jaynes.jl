module BayesianLinearRegression

include("../src/Jaynes.jl")
using .Jaynes

function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = Vector{Float64}(undef, length(x))
    for i in 1 : length(x)
        push!(y, rand(:y => i, Normal(β * x[i], σ)))
    end
    return y
end

x = [Float64(i) for i in 1 : 100]
obs = selection(map(1 : 100) do i
                    (:y => i, ) => 3.0 * x[i] + rand()
                end)

n_samples = 5000
@time ps, ret = importance_sampling(obs, n_samples, bayesian_linear_regression, (x, ))

mean_σ = sum(map(ps.calls) do cl
                 get_ret(cl[:σ])
             end) / n_samples
println("Mean σ: $mean_σ")

mean_β = sum(map(ps.calls) do cl
                 get_ret(cl[:β])
             end) / n_samples
println("Mean β: $mean_β")

end # module
