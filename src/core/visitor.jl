# ------------ Lightweight visitor ------------ #

const Visitor = DynamicTarget

function visit!(vs::Visitor, addr)
    haskey(vs.tree, addr) && error("VisitorError (visit!): already visited address $(addr).")
    push!(vs, addr)
end

function compare(am::AddressMap, vs::Visitor)
    missed = []
    for (k, v) in shallow_iterator(am)
        k in vs || push!(missed, k)
    end
    isempty(missed), missed
end
