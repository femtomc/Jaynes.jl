geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end

cl = Jaynes.call(Trace(), rand, :geo, geo, 0.5)
@test haskey(cl.trace, :geo)
