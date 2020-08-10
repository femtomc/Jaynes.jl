# ------------ Constrain anywhere ------------ #

struct ConstrainedAnywhereSelection <: ConstrainedSelection
    query::Dict{Address, Any}
    ConstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new(Dict{Address, Any}(obs))
end

has_top(cas::ConstrainedAnywhereSelection, addr::T) where T <: Address = haskey(cas.query, addr)
has_top(cas::ConstrainedAnywhereSelection, addr::T) where T <: Tuple = haskey(cas.query, addr[end])

dump_queries(cas::ConstrainedAnywhereSelection) = collect(cas.query)

get_top(cas::ConstrainedAnywhereSelection, addr::T) where T <: Address = getindex(cas.query, addr)
get_top(cas::ConstrainedAnywhereSelection, addr::T) where T <: Tuple = getindex(cas.query, addr[end])

get_sub(cas::ConstrainedAnywhereSelection, addr) = cas

isempty(cas::ConstrainedAnywhereSelection) = isempty(cas.query)

# ------------ Pretty printing ------------ #

function Base.display(chs::ConstrainedAnywhereSelection)
    println("  __________________________________\n")
    println("              Selection\n")
    for a in keys(chs.query)
        println(" (Anywhere)   $(a) : $(chs.query[a])")
    end
    println("  __________________________________\n")
end

# ------------ Unconstrained select anywhere ------------ #

struct UnconstrainedAnywhereSelection{T <: UnconstrainedSelectQuery} <: UnconstrainedSelection
    query::T
    UnconstrainedAnywhereSelection(obs::Vector{Tuple{T, K}}) where {T <: Any, K} = new{UnconstrainedByAddress}(UnconstrainedByAddress(Dict{Address, Any}(obs)))
    UnconstrainedAnywhereSelection(obs::Tuple{T, K}...) where {T <: Any, K} = new{UnconstrainedByAddress}(UnconstrainedByAddress(Dict{Address, Any}(collect(obs))))
end

has_top(cas::UnconstrainedAnywhereSelection, addr::T) where T <: Address = has_top(cas.query, addr)
has_top(cas::UnconstrainedAnywhereSelection, addr::T) where T <: Tuple = has_top(cas.query, addr[end])
dump_queries(cas::UnconstrainedAnywhereSelection) = dump_queries(cas.query)
get_sub(cas::UnconstrainedAnywhereSelection, addr) = cas
isempty(cas::UnconstrainedAnywhereSelection) = isempty(cas.query)

# ------------ Unconstrained select all ------------ #

struct UnconstrainedAllSelection <: UnconstrainedSelection end

has_top(uas::UnconstrainedAllSelection, addr) = true
get_sub(uas::UnconstrainedAllSelection, addr) = uas
isempty(uas::UnconstrainedAllSelection) = false
