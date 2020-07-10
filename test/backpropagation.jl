function learnable_normal(x::Float64, y::Float64)
    l = learnable(:l, 3.0)
    m = learnable(:m, 10.0)
    q = rand(:q, Normal(l, 3 + m^2))
    return q
end

@testset "Convergence for learnable Normal" begin
    ret, cl, w = generate(selection((:q, 6.0)), learnable_normal, 5.0, 3.0)
    params = get_parameters(cl)
    for i in 1:100
        ret, cl, w = generate(selection((:q, 6.0)), learnable_normal, 5.0, 3.0; params = params)
        param_grads = get_parameter_gradients(cl, 1.0)
        update!(params, param_grads)
    end
    @test params.utility[:l] ≈ 6.0 atol = 1e-3
    @test params.utility[:m] ≈ 0.0 atol = 1e-3
end

#function learnable_hypers(x::Float64, y::Float64)
#    l = learnable(:l, 3.0)
#    m = learnable(:m, 10.0)
#    p = rand(:p, Normal(l, 1.0))
#    t = rand(:t, Normal(m, 1.0))
#    q = rand(:q, Normal(p + t, 1.0))
#    return q
#end
#
#@testset "Convergence for learnable hypers" begin
#    cl, w = generate(selection((:q, 6.0)), learnable_hypers, 5.0, 3.0)
#    params = get_parameters(cl)
#    for i in 1:1000
#        cl, w = generate(selection((:q, 6.0)), learnable_hypers, 5.0, 3.0; params = params)
#        param_grads = get_parameter_gradients(cl, 1.0)
#        update!(params, param_grads)
#        display(params; show_values = true)
#    end
#    @test params.utility[:l] ≈ 6.0 atol = 1e-3
#    @test params.utility[:m] ≈ 0.0 atol = 1e-3
#end
