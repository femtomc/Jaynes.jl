# ------------ Constrain anywhere ------------ #

struct ConstrainedAnywhereSelection{T <: ConstrainedSelectQuery} <: ConstrainedSelection
    query::T
    ConstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new{ConstrainedByAddress}(ConstrainedByAddress(Dict{Address, Any}(obs)))
    ConstrainedAnywhereSelection(obs::Tuple{T, K}...) where {T <: Any, K} = new{ConstrainedByAddress}(ConstrainedByAddress(Dict{Address, Any}(collect(obs))))
end

has_query(cas::ConstrainedAnywhereSelection, addr::T) where T <: Address = has_query(cas.query, addr)
has_query(cas::ConstrainedAnywhereSelection, addr::T) where T <: Tuple = has_query(cas.query, addr[end])

dump_queries(cas::ConstrainedAnywhereSelection) = dump_queries(cas.query)

get_query(cas::ConstrainedAnywhereSelection, addr::T) where T <: Address = get_query(cas.query, addr)
get_query(cas::ConstrainedAnywhereSelection, addr::T) where T <: Tuple = get_query(cas.query, addr[end])

get_sub(cas::ConstrainedAnywhereSelection, addr) = cas

isempty(cas::ConstrainedAnywhereSelection) = isempty(cas.query)

# ------------ Pretty printing ------------ #

function Base.display(chs::ConstrainedAnywhereSelection)
    println("  __________________________________\n")
    println("              Selection\n")
    addrs = Union{Symbol, Pair}[]
    chd = Dict{Address, Any}()
    collect!(addrs, chd, chs.query)
    for a in addrs
        println(" (Anywhere)   $(a) : $(chd[a])")
    end
    println("  __________________________________\n")
end

# ------------ Unconstrained select anywhere ------------ #

struct UnconstrainedAnywhereSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    query::T
    UnconstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new{UnconstrainedByAddress}(UnconstrainedByAddress(Dict{Address, Any}(obs)))
    UnconstrainedAnywhereSelection(obs::Tuple{T, K}...) where {T <: Any, K} = new{UnconstrainedByAddress}(UnconstrainedByAddress(Dict{Address, Any}(collect(obs))))
end

has_query(cas::UnconstrainedAnywhereSelection, addr::T) where T <: Address = has_query(cas.query, addr)
has_query(cas::UnconstrainedAnywhereSelection, addr::T) where T <: Tuple = has_query(cas.query, addr[end])
dump_queries(cas::UnconstrainedAnywhereSelection) = dump_queries(cas.query)
get_sub(cas::UnconstrainedAnywhereSelection, addr) = cas
isempty(cas::UnconstrainedAnywhereSelection) = isempty(cas.query)

# ------------ Unconstrained select all ------------ #

struct UnconstrainedAllSelection <: UnconstrainedSelection end

has_query(uas::UnconstrainedAllSelection, addr) = true
get_sub(uas::UnconstrainedAllSelection, addr) = uas
isempty(uas::UnconstrainedAllSelection) = false
