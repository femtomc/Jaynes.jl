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

function target(v::Vector{Pair{T, K}}) where {T <: Tuple, K}
    tg = DynamicMap{Value}()
    for (k, v) in v
        set_sub!(tg, k, Value(v))
    end
    tg
end

# Visitor.
include("core/visitor.jl")

# Learnables and gradients.
include("core/learnables.jl")
