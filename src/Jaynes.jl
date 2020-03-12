module Jaynes

# TODO: kill.
using ExportAll

using MetaGraphs, LightGraphs

using IRTools
using IRTools: blocks, var, delete!, arguments, argument!, insertafter!, isreturn, emptyargs!, branches, renumber
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, Statement, isexpr, return!, func, evalir
using IRTools: Pipe, finish, xcall, deletearg!
using IRTools: prewalk

using Random

using Flux
using Flux: Recur, RNNCell
using Zygote
using Zygote: gradient

using Distributions
using DistributionsAD

using FunctionalCollections

# GPU support.

import Base.rand

include("compiler/Compiler.jl")

@exportAll()

end # Jaynes :)
