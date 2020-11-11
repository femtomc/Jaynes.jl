# ------------ Gradient store ------------ #

struct Store
    params::Dict{Address,Any}
    Store() = new(Dict{Address, Any}())
    Store(d::Dict{Address, Any}) = new(d)
end
haskey(ps::Store, addr) = haskey(ps.params, addr)
setindex!(ps::Store, val, addr) = ps.params[addr] = val
getindex(ps::Store, addr) = ps.params[addr]

Zygote.@adjoint Store(params) = Store(params), store_grad -> (nothing,)

function +(a::Store, b::Store)
    params = Dict{Address, Any}()
    for (k, v) in Iterators.flatten((a.params, b.params))
        if !haskey(params, k)
            params[k] = v
        else
            params[k] += v
        end
    end
    Store(params)
end
