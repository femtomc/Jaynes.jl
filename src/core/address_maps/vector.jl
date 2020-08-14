# ------------ Vector map ------------ #

struct VectorMap{K} <: AddressMap{K}
    vector::Vector{AddressMap{<: K}}
    VectorMap{K}() where K = new{K}(Vector{AddressMap{K}}())
    VectorMap{K}(vector::Vector{<: AddressMap{K}}) where K = new{K}(vector)
    
    # TODO: figure out this type witchcraft.
    VectorMap{K}(vector::Vector{AddressMap{<: T}}) where {K, T} = new{K}(vector)
end
VectorMap(vector::Vector{<: AddressMap{K}}) where K = VectorMap{K}(vector)
Zygote.@adjoint VectorMap(vector) = VectorMap(vector), ret_grad -> (nothing, )

@inline shallow_iterator(vm::VectorMap) = enumerate(vm.vector)

@inline get_sub(vm::VectorMap, addr::A) where A <: Address = get(vm.vector, addr, Empty())
@inline get_sub(vm::VectorMap, addr::Tuple{}) = Empty()

@inline Base.isempty(vm::VectorMap) = isempty(vm.vector)

function haskey(vm::VectorMap, addr)
    addr < length(vm.vector)
end

function push!(vm::VectorMap{K}, v::AddressMap{<:K}) where K
    push!(vm.vector, v)
end

function set_sub!(vm::VectorMap{K}, addr, v::AddressMap{<:K}) where K
    haskey(vm, addr) || error("(set_sub!): field $(:vector) of instance type VectorMap does not have $addr as index.")
    insert!(vm.vector, addr, v)
end
function set_sub!(vm::VectorMap{K}, addr::Tuple{}, v::AddressMap{<:K}) where K end
function set_sub!(vm::VectorMap{K}, addr::Tuple{T}, v::AddressMap{<:K}) where {K, T}
    set_sub!(vm, addr[1], v)
end
function set_sub!(vm::VectorMap{K}, addr::Tuple, v::AddressMap{<:K}) where K
    hd, tl = addr[1], addr[2 : end]
    !haskey(t.vm, hd) && error("(set_sub!): instance of type VectorMap does not have index with head $hd of $addr.")
    set_sub!(t.vector[hd], tl, v)
end

function merge(vm1::VectorMap{K},
               vm2::VectorMap{K}) where K
    vector = AddressMap{K}[]
    if length(vm1) > length(vm2)
        for (k, v) in shallow_iterator(vm2)
            push!(vector, merge(get_sub(vm1, k), v))
        end
        for (_, v) in shallow_iterator(vm1)[length(vm2) : end]
            push!(vector, v)
        end
    else
        for (k, v) in shallow_iterator(vm1)
            push!(vector, merge(v, get_sub(vm2, k)))
        end
        for (_, v) in shallow_iterator(vm2)[length(vm1) : end]
            push!(vector, v)
        end
    end
    vector
end
merge(vm::VectorMap, ::Empty) = deepcopy(vm)
merge(::Empty, vm::VectorMap) = deepcopy(vm)
Zygote.@adjoint merge(a, b) = merge(a, b), s_grad -> (nothing, nothing)
+(a::VectorMap{K}, b::VectorMap{K}) where K = merge(a, b)

function collect!(par::T, addrs::Vector, chd::Dict, vm::VectorMap, meta) where T <: Tuple
    iterate(vm) do (k, v)
        collect!((par..., k), addrs, chd, v, meta)
    end
end
function collect!(addrs::Vector, chd::Dict, vm::VectorMap, meta)
    iterate(vm) do (k, v)
        collect!((k, ), addrs, chd, v, meta)
    end
end
