import Base: getindex, haskey, rand, iterate, isempty
import Base: collect, getindex, setindex!, push!
import Base: +

# Maps.
include("structures/address_map.jl")

# Traces.
include("structures/traces.jl")
Trace() = DynamicTrace()

# Selections.
include("structures/selections.jl")

# Visitor.
include("structures/visitor.jl")

# Learnables and gradients.
include("structures/learnables.jl")
