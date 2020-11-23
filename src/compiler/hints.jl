abstract type CompilerHint end
abstract type ProgramStructureHint <: CompilerHint end
abstract type AddressingHint <: CompilerHint end

include("static/kernel_hint.jl")
include("static/switch_hint.jl")
include("static/dynamic_address_hint.jl")
