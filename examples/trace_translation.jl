module TraceTranslators

include("../src/Jaynes.jl")
using .Jaynes

# Available on master Gen.
using Gen

polar = @jaynes function p1()
    r ~ InvGamma(1, 1)
    θ ~ Uniform(-π / 2, π / 2)
end

cartesian = @jaynes function p2()
    x ~ Normal(0, 1)
    y ~ Normal(0, 1)
end

@transform f (t1) to (t2) begin
    r = @read(t1[:r], :continuous)
    theta = @read(t1[:theta], :continuous)
    @write(t2[:x], r * cos(theta), :continuous)
    @write(t2[:y], r * sin(theta), :continuous)
end

@transform finv (t2) to (t1) begin
    x = @read(t2[:x], :continuous)
    y = @read(t2[:y], :continuous)
    r = sqrt(x^2 + y^2)
    @write(t1[:r], sqrt(x^2 + y^2), :continuous)
    @write(t1[:theta], atan(y, x), :continuous)
end

pair_bijections!(f, finv)
translator = DeterministicTraceTranslator(cartesian, (), choicemap(), f)
tr = simulate(cartesian, ())
t2, log_weight = translator(tr)

end # module
