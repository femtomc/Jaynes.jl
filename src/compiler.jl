# Generic.
include("compiler/utils.jl")
include("compiler/loop_detection.jl")
include("compiler/reaching.jl")
include("compiler/address_blanket.jl")
include("compiler/tracing/tracer.jl")

# Kernel detection and dynamic addressing hints.
include("compiler/hints.jl")

# Trace types system
include("compiler/tracing/trace_types.jl")

# Support error checker.
include("compiler/static/support_checker.jl")

# Partial evaluation/specialization
include("compiler/specializer/diffs.jl")
include("compiler/specializer/transforms.jl")
include("compiler/specializer/interface.jl")

# Automatic addressing converts stochastic functions into PPs
include("compiler/automatic_addressing/automatic_addressing_transform.jl")

# ------------ Documentation ------------ #
