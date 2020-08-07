function elliptical_slice(addr::K, 
                          μ::Vector{Float64}, 
                          Σ, 
                          cl::C) where {K <: Tuple, C <: CallSite}
    ν = rand(MvNormal(zeros(length(μ)), Σ))
    u = rand(Uniform(0.0, 1.0))
    θ = rand(Uniform(0.0, 2*π))
    θ_min, θ_max = θ - 2 * π, θ
    old = cl[addr] .- μ
    new = old * cos(θ) + ν * sin(θ)
    _, cl, w, _, _ = update(selection([addr => new .+ μ]), cl)
    while w <= log(u)
        if θ < 0
            θ_min = θ
        else
            θ_max = θ
        end
        θ = rand(Uniform(θ_min, θ_max))
        new = old * cos(θ) + ν * sin(θ)
        _, cl, w, _, _ = update(selection([addr => new .+ μ]), cl)
    end
    return (cl, true)
end
