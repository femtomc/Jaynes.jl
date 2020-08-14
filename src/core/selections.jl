const Target = AddressMap{<:Union{Empty, Select}}

@inline function Base.in(addr, tg::Target)
    get_sub(tg, addr) === SelectAll()
end

Base.getindex(tg::AddressMap{S}, addr) where S <: Select = get_sub(tg, addr)
get_sub(tg::Target, addr) = get_sub(tg, addr)
Base.merge(::SelectAll, ::Target) = SelectAll()
Base.merge(::Target, ::SelectAll) = SelectAll()
Base.merge(::SelectAll, ::SelectAll) = SelectAll()
Base.merge(::SelectAll, ::Empty) = SelectAll()
Base.merge(::Empty, ::SelectAll) = SelectAll()

const DynamicTarget = DynamicMap{Select}

@inline Base.push!(tg::DynamicTarget, addr) = set_sub!(tg, addr, SelectAll())
@inline function Base.push!(tg::DynamicTarget, addr::Tuple{}) end
@inline function Base.push!(tg::DynamicTarget, addr::Tuple{T}) where T
    Base.push!(s, addr[1])
end
@inline function Base.push!(tg::DynamicTarget, addr::Tuple) where T
    hd, tl = addr
    sub = get_sub(tg, hd)
    if sub isa DynamicTarget
        push!(sub, tl)
    else
        new = target(rest)
        merge!(new, sub)
        set_sub!(tg, hd, new)
    end
end

function target(addrs...)
    tg = DynamicTarget()
    for addr in addrs
        set_sub!(tg, addr, SelectAll())
    end
    tg
end
