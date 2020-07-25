# ------------ Constraints to direct addresses ------------ #

struct ConstrainedByAddress <: ConstrainedSelectQuery
    query::Dict{Any, Any}
    ConstrainedByAddress() = new(Dict())
    ConstrainedByAddress(d::Dict) = new(d)
end

has_query(csa::ConstrainedByAddress, addr) = haskey(csa.query, addr)
dump_queries(csa::ConstrainedByAddress) = keys(csa.query)
get_query(csa::ConstrainedByAddress, addr) = getindex(csa.query, addr)
isempty(csa::ConstrainedByAddress) = isempty(csa.query)
function push!(sel::ConstrainedByAddress, addr, val)
    sel.query[addr] = val
end
function merge!(sel1::ConstrainedByAddress,
                sel2::ConstrainedByAddress)
    Base.merge!(sel1.query, sel2.query)
end
addresses(csa::ConstrainedByAddress) = keys(csa.query)
function compare(chs::ConstrainedByAddress, v::Visitor)
    addrs = []
    for addr in addresses(chs)
        addr in v.addrs && continue
        push!(addrs, addr)
    end
    return isempty(addrs), addrs
end
function filter(k_fn::Function, v_fn::Function, query::ConstrainedByAddress) where T <: Address
    top = ConstrainedByAddress()
    for (k, v) in query.query
        k_fn(k) && v_fn(v) && push!(top, k, v)
    end
    return top
end
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

has_query(csa::UnconstrainedByAddress, addr) = addr in csa.query
dump_queries(csa::UnconstrainedByAddress) = keys(csa.query)
isempty(csa::UnconstrainedByAddress) = isempty(csa.query)
function push!(sel::UnconstrainedByAddress, addr::Symbol)
    push!(sel.query, addr)
end
function push!(sel::UnconstrainedByAddress, addr::Pair{Symbol, Int64})
    push!(sel.query, addr)
end
function push!(sel::UnconstrainedByAddress, addr::Int64)
    push!(sel.query, addr)
end
addresses(usa::UnconstrainedByAddress) = usa.query
function filter(k_fn::Function, v_fn::Function, query::UnconstrainedByAddress) where T <: Address
    top = UnconstrainedByAddress()
    for k in query.query
        k_fn(k) && push!(top, k)
    end
    return top
end
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

