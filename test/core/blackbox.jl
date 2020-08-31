# Geometric.
geo(p::Float64) = rand(:flip, Bernoulli(p)) ? 0 : 1 + rand(:geo, geo, p)

@testset "Trace" begin
    ret, cl = simulate(geo, 0.5)
    @test haskey(cl.trace, :flip)
end

@testset "Generate" begin
    sel = target([(:flip, ) => true])
    ret, cl, w = generate(sel, geo, 0.5)
    @test haskey(cl.trace, :flip)
    @test cl[:flip] == true
end

@testset "Update" begin
    ret, cl = simulate(geo, 0.5)
    sel = target([(:flip, ) => true])
    ret, cl, w, retdiff, d = update(sel, cl)
    @test cl[:flip] == true
end
