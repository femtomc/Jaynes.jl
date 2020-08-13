function piecewise_deterministic_markov_kernel(selection::S, 
                                               cl::C,
                                               Flow, 
                                               θ;
                                               retval_grad = 1.0,
                                               c=1e1, 
                                               λ0=Flow.λref) where {S <: UnconstrainedSelection, C <: CallSite}
    args = get_args(cl)
    sel_values, choice_grads = get_choice_gradients(selection, cl, retval_grad)
    values = array(sel_values, Float64)
    gradient = array(choice_grads, Float64)
    x = values
    ZigZagBoomerang.refresh!(θ, Flow)
    t = 0.0
    ∇ϕx = copy(θ)
    acc = num = 0
    function ∇ϕ!(y, x, sel_values, cl, args, argdiffs, selection, retval_grad)
        values = x
        sel_values[] = from_array(sel_values[], values)
        cl[]  = update(sel_values[], cl[], args, argdiffs)[2]
        (_, _, choice_grads) = choice_gradients(selection, cl[], retval_grad)
        gradient = array(choice_grads, Float64)
        @. y = -gradient
        y
    end
    Ξ = ZigZagBoomerang.Trace(t, x, θ, Flow)
    τref = T = ZigZagBoomerang.waiting_time_ref(Flow)
    a, b = ZigZagBoomerang.ab(x, θ, c, Flow)
    t′ = t + poisson_time(a, b, rand())
    while t < T
        t, x, θ, (acc, num), c, a, b, t′, τref = ZigZagBoomerang.pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, a, b, t′, τref, (acc, num), Flow, Ref(sel_values), Ref(cl), args, argdiffs, selection, retval_grad; adapt=false)
    end
    values = x
    cl = cl
    sel_values = from_array(sel_values, values)
    ret, cl, _= update(sel_values, cl, args, argdiffs)
    return (cl, true)
end
