module SelectionCreation

include("../src/Jaynes.jl")
using .Jaynes
using .Jaynes: Address, selection

unc_vec = Union{Symbol, Pair}[:x => :y, :x => 2, :z => :q => :x, :m]
con_vec = Tuple{Union{Symbol, Pair}, Any}[(:x => :y, 5.0), (:x => 2, 6.0), (:z => :q => :x, 7.0)]

println(selection(unc_vec))
println(selection(con_vec))

end #module
