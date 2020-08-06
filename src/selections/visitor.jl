# ------------ Lightweight visitor ------------ #

struct Visitor <: Selection
    addrs::Vector{Any}
    Visitor() = new([])
end

push!(vs::Visitor, addr::Address) = push!(vs.addrs, addr)
has_top(vs::Visitor, addr::Address) = addr in vs.addrs
has_sub(vs::Visitor, addr::Address) = addr in vs.addrs

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
