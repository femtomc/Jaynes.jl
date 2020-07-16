# ------------ Lightweight visitor ------------ #

struct Visitor <: Selection
    addrs::Vector{Union{Symbol, Pair}}
    Visitor() = new(Union{Symbol, Pair}[])
end

push!(vs::Visitor, addr::Address) = push!(vs.addrs, addr)

function visit!(vs::Visitor, addr)
    addr in vs.addrs && error("VisitorError (visit!): already visited address $(addr).")
    push!(vs, addr)
end

function visit!(vs::Visitor, addrs::Vector)
    for addr in addrs
        addr in vs.addrs && error("VisitorError (visit!): already visited address $(addr).")
        push!(vs, addr)
    end
end

function visit!(vs::Visitor, par::Address, addrs::Vector)
    for addr in addrs
        addr in vs.addrs && error("VisitorError (visit!): already visited address $(addr).")
        push!(vs, par => addr)
    end
end

function set_sub!(vs::Visitor, addr::Address, sub::Visitor)
    haskey(vs.tree, addr) && error("VisitorError (set_sub!): already visited address $(addr).")
    vs.tree[addr] = sub
end

function has_sub(vs::Visitor, addr::Address)
    return haskey(vs.tree, addr)
end

function get_sub(vs::Visitor, addr::Address)
    haskey(vs.tree, addr) && return vs.tree[addr]
    error("VisitorError (get_sub): sub not defined at $addr.")
end

function has_query(vs::Visitor, addr::Address)
    return addr in vs.addrs
end

isempty(vs::Visitor) = isempty(vs.tree) && isempty(vs.addrs)

