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
tr = simulate(jfunc, (5, ))
display(tr)

# ------------ Gibbs ------------ #

@kern function gibbs_kernel(trace)
    trace ~ mh(trace, select(:n))
    trace ~ mh(trace, select(:λ₁))
    trace ~ mh(trace, select(:λ₂))
end

# ------------ Inference ------------ #

infer = () -> begin
    obs = target([(:x => 1, ) => 1,
                  (:x => 2, ) => 2,
                  (:x => 3, ) => 3,
                  (:x => 4, ) => 12,
                  (:x => 5, ) => 10])

    # Sample trace.
    tr, w = generate(jfunc, (5, ), obs)
    display(tr)

    trs = Trace[]
    for i in 1 : 2000
        tr, _ = gibbs_kernel(tr)
        i % 50 == 0 && push!(trs, tr)
    end

    # Posterior est.
    λ₁_est, λ₂_est, n_est = zip(map(trs) do tr
                                    (tr[:λ₁], tr[:λ₂], tr[:n])
                                end...)
    println("est: $((sum(λ₁_est) / 50.0, 
                     sum(λ₂_est) / 50.0,
                     sum(n_est) / 50.0))")

end
infer()

end # module
