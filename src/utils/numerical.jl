# Log sum exp.
function lse(arr)
    max = maximum(arr)
    max == -Inf ? -Inf : max + log(sum(exp.(arr .- max)))
end

function lse(x1::Real, x2::Real)
    m = max(x1, x2)
    m == -Inf ? m : m + log(exp(x1 - m) + exp(x2 - m))
end

# Effective sample size.
function ess(lnw::Vector{Float64})
    log_ess = -lse(2. * lnw)
    return exp(log_ess)
end

# Normalize log weights.
function nw(lw::Vector{Float64})
    lt = lse(lw)
    lnw = lw .- lt
    return (lt, lnw)
end
