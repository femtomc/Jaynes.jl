# Model.
function bayesian_linear_regression(x::Vector{Float64})
    σ = rand(:σ, InverseGamma(2, 3))
    β = rand(:β, Normal(1.0, 3.0))
    y = [rand(:y => i, Normal(β * x[i], σ)) for i in 1 : length(x)]
    y
end
