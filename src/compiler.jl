function lower_to_ir(call, argtypes...)
    sig = length(argtypes) == 1 && argtypes[1] == Tuple{} ? begin
        Tuple{typeof(call)}
    end : Tuple{typeof(call), argtypes...}
    m = meta(sig)
    ir = IR(m)
    return ir
end

@inline control_flow_check(ir) = !(length(ir.blocks) > 1)

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
