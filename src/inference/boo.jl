function boomerang(sel::K, 
                   cl::C) where {K <: UnconstrainedSelection, C <: CallSite}

    p_mod_score = get_score(cl)
    sel_values, choice_grads = get_choice_gradients(sel, cl, 1.0)

    # TODO: re-factor.
    sel_values_ref = Ref(sel_values)
    cl_ref = Ref(cl)

    x = array(sel_values, Float64)
    d = length(x)
    Flow = Boomerang(sparse(I, d, d), zeros(d), 1.0)
    N = MvNormal(d, 1.0)
    θ = rand(N)
    t = 0.0
    ∇ϕx = copy(θ)
    acc = num = 0
    function ∇ϕ!(y, x, sel, cl_ref, sel_values_ref)
        sel_values_ref[] = selection(sel_values_ref[], x)
        ret, cl_ref[], _ = update(sel_values_ref[], cl_ref[])[2]
        sel_values_ref[], choice_grads = get_choice_gradients(sel, cl_ref[], 1.0)
        y .= array(choice_grads, Float64)
    end

    Ξ = ZigZagBoomerang.Trace(t, x, θ, Flow) # should persist between calls
    τref = T = ZigZagBoomerang.waiting_time_ref(Flow)
    c = 100.
    a, b = ZigZagBoomerang.ab(x, θ, c, Flow)
    t′ = t + poisson_time(a, b)
    while t < T
        t, x, θ, (acc, num), c, a, b, t′, τref = ZigZagBoomerang.pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, a, b, t′, τref, (acc, num), Flow, sel, cl_ref, sel_values_ref; adapt=false)
    end
    sel_values = selection(sel_values_ref[], x)
    ret, cl, w = update(sel_values, cl_ref[])
    (cl, true)
end
