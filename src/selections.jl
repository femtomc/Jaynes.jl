import Base: haskey, getindex, push!, merge!, union, isempty, merge
import Base.collect
import Base.filter
import Base: +

# ------------ Selection ------------ #

# Abstract type for a sort of query language for addresses within a particular method body.
abstract type Selection end

# ------------ Generic conversion to and from arrays ------------ #

# Subtypes of Selection should implement their own fill_array! and from_array.
# Right now, these methods are really only used for gradient-based learning.

function array(gs::K, ::Type{T}) where {T, K}
    arr = Vector{T}(undef, 32)
    n = fill_array!(gs, arr, 1)
    resize!(arr, n)
    arr
end

function fill_array!(val::T, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind
        resize!(arr, 2 * f_ind)
    end
    arr[f_ind] = val
    1
end

function fill_array!(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    if length(arr) < f_ind + length(val)
        resize!(arr, 2 * (f_ind + length(val)))
    end
    arr[f_ind : f_ind + length(val) - 1] = val
    length(val)
end

function selection(schema::K, arr::Vector) where K
    (n, sel) = from_array(schema, arr, 1)
    n != length(arr) && error("Dimension error: length of arr $(length(arr)) must match $n.")
    sel
end

function from_array(::T, arr::Vector{T}, f_ind::Int) where T
    (1, arr[f_ind])
end

function from_array(val::Vector{T}, arr::Vector{T}, f_ind::Int) where T
    n = length(val)
    (n, arr[f_ind : f_ind + n - 1])
end

# ------------ Visitor ------------ #

include("selections/visitor.jl")

# ------------ Constrained and unconstrained selections ------------ #

abstract type ConstrainedSelection end
abstract type UnconstrainedSelection end
abstract type SelectQuery <: Selection end
abstract type ConstrainedSelectQuery <: SelectQuery end
abstract type UnconstrainedSelectQuery <: SelectQuery end

# Note: there are selections and "query" structs which can be used into selections.

include("selections/empty.jl")
include("selections/by_address.jl")
include("selections/anywhere.jl")
include("selections/hierarchical.jl")
include("selections/union.jl")

# ------------ Convenience constructors ------------ #

selection() = ConstrainedEmptySelection()
anywhere() = ConstrainedEmptySelection()
union() = ConstrainedEmptySelection()
intersection() = ConstrainedEmptySelection()

# By address
function selection(a::Vector{K}) where K <: Tuple
    return UnconstrainedHierarchicalSelection(a)
end

function selection(a::Vector{Pair{K, J}}) where {K <: Tuple, J}
    return ConstrainedHierarchicalSelection(a)
end

# Anywhere
function anywhere(a::Vector{Pair{K, J}}) where {K <: Tuple, J}
    new = map(a) do (k, v)
        !(k isa Tuple{Address}) && error("Anywhere construction requires that you only use addresses of type Address.")
        (k[1], v)
    end
    return ConstrainedAnywhereSelection(new)
end

function anywhere(a::Vector{T}) where T <: Tuple
    new = map(a) do k
        !(k isa Tuple{Address}) && error("Anywhere construction requires that you only use addresses of type Address.")
        k[1]
    end
    return UnconstrainedAnywhereSelection(new)
end

# ------------ Set operations on selections ------------ #

union(a...) = ConstrainedUnionSelection(ConstrainedSelection[a...])
function intersection(a::ConstrainedSelection...) end

# ------------ Documentation ------------ #

@doc(
"""
```julia
constraints = selection(a::Vector{Pair{K, J}}) where {K <: Tuple, J}
targets = selection(a::Vector{K}) where K <: Tuple
```

The first version of the `selection` API function will produce a `ConstrainedHierarchicalSelection` with a direct address-based query called `ConstrainedByAddress`. Here's an example of use:

```julia
selection([(:x, ) => 10.0,
           (:y, :z => 5) => 6.0])
```

In the call, the syntax is that addresses at which calls occur are separated by commas. So `(:x, )` is just an address in the top-level call and `(:y, :z => 5)` is an address in the call specified at address `:y`.

The second version produces an `UnconstrainedHierarchicalSelection` with a direct address-based unconstrained query called `UnconstrainedByAddress`.

```julia
selection([(:x, ), (:y, :z)])
```

The exact same syntax considerations apply. Unconstrained addressing is used for incremental inference (i.e. methods which use `RegenerateContext` like Markov chain Monte Carlo).
""", selection)

@doc(
"""
```julia
anywhere_selection = anywhere(a::Vector{Pair{K, J}}) where {K <: Tuple, J}
```

The `anywhere` selection API provides access to a special set of constrained and unconstrained selections which will apply at any level of the call stack. Usage of this API requires that you pass in a similar structure to `selection` - with the caveat that you can only use top-level addresses in the tuple:

```julia
anywhere([(:x, ) => 5.0, (:y => 10, ) => 10.0])
```

The interpretation: at any place in the call stack where the end of the address is equal to one of the addresses in the selection, the tracer will constrain the address.

There is also an unconstrained version of `anywhere` which provides the same interpretation, but for unconstrained targets of `RegenerateContext` and MCMC algorithms.
""", anywhere)
