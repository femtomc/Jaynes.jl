# ------------ Simple gradients ------------ #

fn1 = () -> begin
    m = learnable(:m)
    q = rand(:q, Normal(m, 3.0))
    q
end

fn2 = () -> begin
    q = rand(:q, Normal(1.0, 3.0))
    q
end

grad_test_model = () -> begin
    N = 10
    q = rand(:fn1, fn1)
    z = rand(:fn2, fn2)
    z + q
end

@testset "Choice gradients" begin
    params = learnables([(:fn1, :m) => 5.0])
    ret, cl = simulate(params, grad_test_model)
    tg = target([(:fn2, :q)])
    vals, as, cgs = get_choice_gradients(tg, params, cl, 1.0)
    @test haskey(cgs, (:fn2, :q))
    @test as == ()
    @test haskey(vals, (:fn2, :q))
end

@testset "Learnable parameter gradients" begin
    params = learnables([(:fn1, :m) => 5.0])
    ret, cl = simulate(params, grad_test_model)
    vals, pgs = get_learnable_gradients(params, cl, 1.0)
    @test haskey(pgs, (:fn1, :m))
end

# ------------ Vectorized gradients ------------ #
