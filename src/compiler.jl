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
