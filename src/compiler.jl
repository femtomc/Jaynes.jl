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

include("compiler/reaching.jl")
include("compiler/address_blanket.jl")
include("compiler/map_codegen.jl")
include("compiler/diffs.jl")
include("compiler/transforms.jl")
include("compiler/interface.jl")

# ------------ Documentation ------------ #
