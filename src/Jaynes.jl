module Jaynes

# TODO: kill.
using ExportAll

using JSON
using MetaGraphs, LightGraphs

using MacroTools

using IRTools
using IRTools: blocks, var, delete!, arguments, argument!, insertafter!, isreturn, emptyargs!, branches, renumber, self
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, Statement, isexpr, return!, func, evalir
using IRTools: Pipe, finish, xcall, deletearg!
using IRTools: prewalk

using Random

using Flux
using Flux: Recur, RNNCell
using Zygote
using Zygote: gradient

using InteractiveUtils: subtypes, @code_lowered
using Distributions

# Get all distributions in Distributions.jl
typesofdist = subtypes(Distribution)
typesofdist2 = filter(x->isa(x,Vector{Any}),subtypes.(typesofdist))
for t in typesofdist2
   global typesofdist # if in REPL
   typesofdist = union(typesofdist, t)
end
dists = map(x -> Base.nameof(x), typesofdist)

using DistributionsAD

using FunctionalCollections

# GPU support.

import Base.rand

include("compiler/Compiler.jl")

@exportAll()

end # Jaynes :)
