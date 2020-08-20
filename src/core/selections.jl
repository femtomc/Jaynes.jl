const Target = AddressMap{<:Union{Empty, Select}}

@inline function Base.in(addr, tg::Target)
    get_sub(tg, addr) === SelectAll()
end

@inline Base.getindex(tg::AddressMap{S}, addr) where S <: Select = get_sub(tg, addr)
@inline get_sub(tg::Target, addr) = get_sub(tg, addr)
@inline Base.merge(::SelectAll, ::Target) = SelectAll()
@inline Base.merge(::Target, ::SelectAll) = SelectAll()
@inline Base.merge(::SelectAll, ::SelectAll) = SelectAll()
@inline Base.merge(::SelectAll, ::Empty) = SelectAll()
@inline Base.merge(::Empty, ::SelectAll) = SelectAll()

const DynamicTarget = DynamicMap{Select}

@inline haskey(dm::DynamicTarget, addr::A) where A <: Address = haskey(dm.tree, addr) && !isempty(get_sub(dm, addr))

Base.merge!(dt::DynamicTarget, s::SelectAll) = dt

@inline Base.push!(tg::DynamicTarget, addr) = set_sub!(tg, addr, SelectAll())
@inline function Base.push!(tg::DynamicTarget, addr::Tuple{}) end
@inline function Base.push!(tg::DynamicTarget, addr::Tuple{T}) where T
    Base.push!(tg, addr[1])
end
@inline function Base.push!(tg::DynamicTarget, addr::Tuple) where T
    hd, tl = addr[1], addr[2 : end]
    sub = get_sub(tg, hd)
    if sub isa DynamicTarget
        push!(sub, tl)
    else
        new = target(tl)
        merge!(new, sub)
        set_sub!(tg, hd, new)
    end
end

function target(v::Vector{T}) where T <: Tuple
    tg = DynamicTarget()
    for k in v
        push!(tg, k)
    end
    tg
end

function target(k::A) where A <: Address
    tg = DynamicTarget()
    push!(tg, k)
    tg
end

function target(k::Tuple)
    tg = DynamicTarget()
    push!(tg, k)
    tg
end
