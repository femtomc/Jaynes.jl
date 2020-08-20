import Base: rand
import Base: getindex, haskey, iterate, isempty, convert, collect, getindex, setindex!, push!, merge, merge!, get, filter
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
