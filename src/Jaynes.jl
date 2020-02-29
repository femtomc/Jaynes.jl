module Jaynes

# TODO: kill.
using ExportAll

using MetaGraphs, LightGraphs
using IRTools
using IRTools: blocks
using IRTools: @code_ir, @dynamo, IR, recurse!, Variable, Statement, isexpr
using Random
using Zygote
using Distributions
using CuArrays
using CUDAnative

import Base.rand

include("static_analysis.jl")
include("inference_compiler.jl")
include("proposal_transform.jl")

@exportAll()

end # Jaynes :)
