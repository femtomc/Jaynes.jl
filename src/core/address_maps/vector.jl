# ------------ Vector map ------------ #

struct VectorMap{K} <: AddressMap{K}
    vector::Vector{AddressMap{<:K}}
    VectorMap{K}() where K = new{K}(Vector{AddressMap{<:K}}())
end

# Vector map interfaces
push!(dm::VectorMap{K}, v::AddressMap{<:K}) where K = push!(dm.vector, v)

function haskey(dm::VectorMap, addr::Int)
    addr <= length(dm.vector) && !isempty(dm.vector[addr])
end
function getindex(dm::VectorMap, addr)
    val = dm.vector[addr]
    isempty(val) ? Empty() : unwrap(val)
end
function setindex!(dm::VectorMap{K}, val::AddressMap{<:K}, addr) where K
    if addr > length(dm.vector) 
        for i in 1 : (addr - length(dm.vector))
            dm.vector[i] = Empty()
        end
    end
    dm.vector[addr] = v
end
function get(dm::VectorMap{K}, addr, val) where K
    haskey(dm, addr) && return getindex(dm, addr)
    return val
end
function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorMap, meta) where T <: Tuple
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, (par..., k))
            chd[(par..., k)] = v.val
        else
            collect!((par..., k), addrs, chd, v.trace, meta)
        end
    end
end
function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorMap, meta)
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, (k, ))
            chd[k] = v.val
        else
            collect!((k, ), addrs, chd, v.trace, meta)
        end
    end
end
