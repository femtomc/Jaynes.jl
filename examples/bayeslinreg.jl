module BayesianLinearRegression

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

function bayesian_linear_regression(N::Int)
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(0.0, 1.0))
    y = plate(:y, x -> rand(:draw, Normal(β*x, σ)), [Float64(i) for i in 1 : N])
end

# Example trace.
ret, cl = simulate(bayesian_linear_regression, 10)
display(cl.trace)

sel = selection(map(1 : 100) do i
                    (:y => i => :draw, Float64(i) + rand())
                end)

@time ps, ret = importance_sampling(sel, 5000, bayesian_linear_regression, (100, ))

num_res = 1000
resample!(ps, num_res)
mean_σ = sum(map(ps.calls) do cl
                 cl[:σ]
             end) / num_res
println("Mean σ: $mean_σ")

mean_β = sum(map(ps.calls) do cl
                 cl[:β]
             end) / num_res
println("Mean β: $mean_β")

end # module
