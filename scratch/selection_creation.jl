module SelectionCreation

include("../src/Jaynes.jl")
using .Jaynes
using .Jaynes: Address

vec = Union{Symbol, Pair}[:x => :y, :x => 2, :z => :q => :x, :m]

struct Pseudotree
    tree::Dict{Address, Pseudotree}
    select::Vector{Address}
    Pseudotree() = new(Dict{Address, Pseudotree}(), Vector{Address}())
end

import Base.push!
function push!(tr::Pseudotree, addr::Symbol)
    push!(tr.select, addr)
end
function push!(tr::Pseudotree, addr::Pair{Symbol, Int64})
    push!(tr.select, addr)
end

function push!(tr::Pseudotree, addr::Pair)
    if !(haskey(tr.tree, addr[1]))
        new = Pseudotree()
        push!(new, addr[2])
        tr.tree[addr[1]] = new
    else
        push!(tr[addr[1]], addr[2])
    end
end

tr = Pseudotree()
for i in vec
    push!(tr, i)
end
println(tr)

end #module
