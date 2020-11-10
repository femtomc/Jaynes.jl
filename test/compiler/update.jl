fn0 = (q, z) -> begin
    m = trace(:m, Normal(q, 1.0))
    q = trace(:q, Normal(z, 5.0))
    m + q
end

fn1 = (q, z, m) -> begin
    t = trace(:z, Normal(5.0, 1.0))
    n = trace(:m, Normal(z, 1.0))
    l = trace(:l, fn0, q, m)
    if l > 5.0
        trace(:q, Normal(1.0, 5.0))
    else
        trace(:q, Normal(5.0, 3.0))
    end
    p = trace(:p, Normal(z, 3.0))
    t = trace(:t, Normal(p, 5.0))
    l + n
end

@testset begin "Dynamic specialization update test 1"
    ret, cl = simulate(fn1, 10.0, 10.0, 1.0)
    prev_score = get_score(cl)

    ret, cl, w, _ = update(Jaynes.Empty(), cl, 
                           Δ(5.0, ScalarDiff(-5.0)), 
                           Δ(5.0, ScalarDiff(-5.0)), 
                           Δ(1.0, NoChange()))
    @test get_score(cl) - w ≈ prev_score
end
