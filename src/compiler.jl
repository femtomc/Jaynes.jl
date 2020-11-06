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

# Support error checker.
include("compiler/support_checker.jl")

# Partial evaluation/specialization
include("compiler/reaching.jl")
include("compiler/address_blanket.jl")
include("compiler/specializer/diffs.jl")
include("compiler/specializer/transforms.jl")
include("compiler/specializer/interface.jl")

# Jaynesizer converts stochastic functions into PPs
include("compiler/jaynesizer/utils.jl")
include("compiler/jaynesizer/jaynesize_transform.jl")

# Trace types system
include("compiler/trace_types/trace_types.jl")

# ------------ Documentation ------------ #
