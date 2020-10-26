struct Σ{N, D}
    weights::SVector{N, Float64}
    cat::Categorical
    components::SVector{N, D}
end
function Mixture(t::Tuple{Float64, D}...) where D <: Distribution
    weights, components = zip(t...)
    Σ(SVector(weights), Categorical(weights...), SVector(components))
end

function (m::Σ)()
    ind = rand(m.cat)
    rand(m.components[ind])
end

@primitive function logpdf(m::Σ, ret)
    sum(log.(m.weights)) + sum(map(m.components) do c
                                 logpdf(c, ret)
                             end)
end
