# Geometric.
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)

# Define as primitive.
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end

@testset "Trace" begin
    cl = trace(geo, 0.5)
    @test haskey(cl.trace, :flip)
end

@testset "Generate" begin
    cl, w = generate(geo, 0.5)
    @test haskey(cl.trace, :flip)
end

@testset "Update" begin
    cl, w = generate(geo, 0.5)
    sel = selection((:flip, true))
    display(sel; show_values = true)
    cl, diff, w = update(sel, cl)
    display(cl)
    @test cl[:flip] == true
    cl, diff, w = update(sel, cl, 0.1)
    @test cl[:flip] == true
end
