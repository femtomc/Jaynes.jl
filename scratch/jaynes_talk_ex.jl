module JaynesTalkExample

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

geo(p::Float64) = rand(:flip, Bernoulli(p)) == 1 ? 0 : 1 + rand(:geo, geo, p)

# Defines a black-box primitive for tracing - will not recurse into this.
Jaynes.@primitive function logpdf(fn::typeof(geo), p, n::Int)
    return Distributions.logpdf(Geometric(p), n)
end

function wats_p()
    p = rand(:p, Beta(1, 1))
    q = rand(:q, geo, p)
    return q
end

function good_prop(q::Int)
    c = ceil(q)
    p = rand(:p, Beta(1, c))
end

obs = Jaynes.selection((:q, 100))
calls, lnw, lmle = Jaynes.importance_sampling(wats_p, (), good_prop, (50, ); observations = obs, num_samples = 50000)

# Show a trace.
display(calls[1].trace; show_values = true)

# Expectation over samples.
println(sum(map(calls) do c
                c[:p]
            end) / 50000)

end #module
