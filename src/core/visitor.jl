# ------------ Lightweight visitor ------------ #

const Visitor = DynamicMap{Empty}

function visit!(vs::Visitor, addr)
    haskey(vs.tree, addr) && error("VisitorError (visit!): already visited address $(addr).")
    set_sub!(vs, addr, Empty())
end
