function lower_to_ir(call, type...)
    sig = Tuple{typeof(call), type...}
    m = meta(sig)
    ir = IR(m)
    return ir
end

function control_flow_check(ir)
    length(ir.blocks) > 1 && return false
    return true
end

# ------------ includes ------------ #

# Partial evaluation/specialization
include("compiler/specializer/reaching.jl")
include("compiler/specializer/address_blanket.jl")
include("compiler/specializer/map_codegen.jl")
include("compiler/specializer/diffs.jl")
include("compiler/specializer/transforms.jl")
include("compiler/specializer/interface.jl")

# Jaynesizer converts stochastic functions into PPs
include("compiler/jaynesizer/utils.jl")
include("compiler/jaynesizer/transform.jl")

# ------------ Documentation ------------ #
