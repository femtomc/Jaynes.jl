import Base: getindex, haskey, rand, iterate, isempty
import Base: collect, getindex, setindex!, push!
import Base: +

# Maps.
include("core/address_map.jl")

# Selections.
include("core/selections.jl")

# Visitor.
include("core/visitor.jl")

# Traces.
include("core/traces.jl")
Trace() = DynamicTrace()

# Learnables and gradients.
include("core/learnables.jl")
