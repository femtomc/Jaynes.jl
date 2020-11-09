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

# Generic.
include("compiler/utils.jl")
include("compiler/loop_detection.jl")
include("compiler/reaching.jl")
include("compiler/address_blanket.jl")

# Kernel detection and dynamic addressing hints.
include("compiler/hints.jl")

# Trace types system
include("compiler/static/trace_types.jl")

# Support error checker.
include("compiler/static/support_checker.jl")

# Partial evaluation/specialization
include("compiler/specializer/diffs.jl")
include("compiler/specializer/transforms.jl")
include("compiler/specializer/interface.jl")

# Jaynesizer converts stochastic functions into PPs
include("compiler/jaynesizer/jaynesize_transform.jl")

# ------------ Documentation ------------ #
