# ------------ Constraints to direct addresses ------------ #

struct ConstrainedByAddress <: ConstrainedSelectQuery
    query::Dict{Any, Any}
    ConstrainedByAddress() = new(Dict())
    ConstrainedByAddress(d::Dict) = new(d)
end

has_top(csa::ConstrainedByAddress, addr) = haskey(csa.query, addr)

dump_queries(csa::ConstrainedByAddress) = Set{Any}(keys(csa.query))

get_top(csa::ConstrainedByAddress, addr) = getindex(csa.query, addr)

isempty(csa::ConstrainedByAddress) = isempty(csa.query)

push!(sel::ConstrainedByAddress, addr, val) = sel.query[addr] = val

merge!(sel1::ConstrainedByAddress, sel2::ConstrainedByAddress) = Base.merge!(sel1.query, sel2.query)

addresses(csa::ConstrainedByAddress) = keys(csa.query)

function compare(chs::ConstrainedByAddress, v::Visitor)
    addrs = []
    for addr in addresses(chs)
        addr in v.addrs && continue
        push!(addrs, addr)
    end
    return isempty(addrs), addrs
end

function ==(cba1::ConstrainedByAddress, cba2::ConstrainedByAddress)
    for (k, v) in cba1.query
        k in keys(cba2.query) || return false
        cba2.query[k] == v || return false
    end
    return true
end

# Functional filter.
function filter(k_fn::Function, v_fn::Function, query::ConstrainedByAddress) where T <: Address
    top = ConstrainedByAddress()
    for (k, v) in query.query
        k_fn(k) && v_fn(v) && push!(top, k, v)
    end
    return top
end

# ------------ Utility for pretty printing ------------ #

function collect!(par, addrs, chd, query::ConstrainedByAddress)
    for (k, v) in query.query
        push!(addrs, (par..., k))
        chd[(par..., k)] = v
    end
end

function collect!(addrs, chd, query::ConstrainedByAddress)
    for (k, v) in query.query
        push!(addrs, (k, ))
        chd[(k, )] = v
    end
end

# ----------- Selection to direct addresses ------------ #

struct UnconstrainedByAddress <: UnconstrainedSelectQuery
    query::Vector{Any}
    UnconstrainedByAddress() = new(Any[])
end

has_top(csa::UnconstrainedByAddress, addr) = addr in csa.query

dump_queries(csa::UnconstrainedByAddress) = csa.query

isempty(csa::UnconstrainedByAddress) = isempty(csa.query)

push!(sel::UnconstrainedByAddress, addr::T) where T <: Address = push!(sel.query, addr)

merge!(sel1::UnconstrainedByAddress, sel2::UnconstrainedByAddress) = union(sel1.query, sel2.query)

addresses(usa::UnconstrainedByAddress) = usa.query

function compare(uba::UnconstrainedByAddress, v::Visitor)
    addrs = []
    for addr in addresses(uba)
        addr in v.addrs && continue
        push!(addrs, addr)
    end
    return isempty(addrs), addrs
end

function ==(cba1::UnconstrainedByAddress, cba2::UnconstrainedByAddress)
    cba1.query == cba2.query
end

# Functional filter.
function filter(k_fn::Function, v_fn::Function, query::UnconstrainedByAddress) where T <: Address
    top = UnconstrainedByAddress()
    for k in query.query
        k_fn(k) && push!(top, k)
    end
    return top
end

# ------------ Utility for pretty printing ------------ #

function collect!(par, addrs, query::UnconstrainedByAddress)
    for k in query.query
        push!(addrs, (par..., k))
    end
end

function collect!(addrs, query::UnconstrainedByAddress)
    for k in query.query
        push!(addrs, (k, ))
    end
end

