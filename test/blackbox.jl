# Geometric.
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)

# Define as primitive.
@primitive function logpdf(fn::typeof(geo), p, count)
    return Distributions.logpdf(Geometric(p), count)
end

@testset "Trace" begin
    ret, cl = simulate(geo, 0.5)
    @test haskey(cl.trace, :flip)
end

@testset "Generate" begin
    sel = selection([(:flip, ) => true])
    ret, cl, w = generate(sel, geo, 0.5)
    @test haskey(cl.trace, :flip)
    @test get_ret(cl[:flip]) == true
end

@testset "Update" begin
    ret, cl = simulate(geo, 0.5)
    sel = selection([(:flip, ) => true])
    ret, cl, w, retdiff, d = update(sel, cl)
    @test get_ret(cl[:flip]) == true
    ret, cl, w, retdiff, d = update(sel, cl, UndefinedChange(), 0.1)
    @test get_ret(cl[:flip]) == true
end
