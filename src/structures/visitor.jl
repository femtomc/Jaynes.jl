# ------------ Lightweight visitor ------------ #

const Visitor = DynamicMap{Empty}

function visit!(vs::Visitor, addr)
    haskey(vs.tree, addr) && error("VisitorError (visit!): already visited address $(addr).")
    push!(vs, addr, Empty())
end
