module ChangePoint

include("../src/Jaynes.jl")
using .Jaynes
using Gen

# ------------ Change point model ------------ #

jfunc = @jaynes function change_point_model(N::Int)
    n ~ DiscreteUniform(N)
    λ₁ ~ Gamma(2, 2)
    λ₂ ~ Gamma(3, 2)
    x = [i >= n ? {:x => i} ~ Poisson(λ₂) : {:x => i} ~ Poisson(λ₁) for i in 1 : N]
    x
end

# Sample trace.
tr = simulate(jfunc, (50, ))
display(tr)

# ------------ Gibbs ------------ #

@kern function gibbs_kernel(trace)
    trace ~ mh(trace, select(:n))
    trace ~ mh(trace, select(:λ₁))
    trace ~ mh(trace, select(:λ₂))
end

# ------------ Inference ------------ #

infer = () -> begin
    obs = target([i <= 30 ? (:x => i, ) => rand(Poisson(3)) : (:x => i, ) => rand(Poisson(7)) for i in 1 : 50])

    # Sample trace.
    tr, w = generate(jfunc, (50, ), obs)
    display(tr)

    trs = Trace[]
    for i in 1 : 4000
        tr, _ = gibbs_kernel(tr)
        i % 50 == 0 && push!(trs, tr)
    end
    display(tr)

    # Posterior est.
    λ₁_est, λ₂_est, n_est = zip(map(trs) do tr
                                    (tr[:λ₁], tr[:λ₂], tr[:n])
                                end...)
    len = length(λ₁_est)
    println("est: $((sum(λ₁_est) / len,
                     sum(λ₂_est) / len,
                     sum(n_est) / len))")

end
infer()

end # module
