import Base: getindex, haskey, rand, iterate, isempty
import Base: collect, getindex, setindex!, push!
import Base: +

# Maps.
include("core/address_map.jl")

# Traces.
include("core/traces.jl")
Trace() = DynamicTrace()

# Selections.
include("core/selections.jl")

# Visitor.
include("core/visitor.jl")

# Learnables and gradients.
include("core/learnables.jl")
