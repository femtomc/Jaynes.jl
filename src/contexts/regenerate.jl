mutable struct RegenerateContext{C <: AddressMap,
                                 T <: AddressMap, 
                                 K <: AddressMap,
                                 D <: AddressMap,
                                 P <: AddressMap} <: ExecutionContext
    prev::C
    tr::T
    target::K
    weight::Float64
    score::Float64
    discard::D
    visited::Visitor
    params::P
end

@inline function record_cached!(ctx::RegenerateContext, addr)
    visit!(ctx, addr)
    sub = get_sub(ctx.prev, addr)
    sc = get_score(sub)
    ctx.score += get_score(sub)
    set_sub!(ctx.tr, addr, sub)
    get_value(sub)
end

function Regenerate(target::K, ps, cl::C, tr, discard) where {K <: AddressMap, C <: CallSite}
    RegenerateContext(cl, 
                      tr,
                      target, 
                      0.0, 
                      0.0, 
                      discard,
                      Visitor(), 
                      Empty())
end

# This uses a "sneaky invoke" hack to allow passage of diffs into user-defined functions whose argtypes do not allow it.
@dynamo function (mx::RegenerateContext{C, T, K})(f, ::Type{S}, args...) where {S <: Tuple, C, T, K}

    # Check for primitive.
    ir = IR(f, S.parameters...)
    ir == nothing && return

    # Equivalent to static DSL optimizations.
    if K <: DynamicMap

        # Release IR normally.
        jaynesize_transform!(ir)
        ir = recur(ir)
        argument!(ir, at = 2)
        ir = renumber(ir)
    else

        # Argument difference inference.
        tr = diff_inference(f, S.parameters, args)

        # Dynamic specialization transform.
        ir = optimization_pipeline(ir.meta, tr, get_address_schema(K))
       
        # Automatic addressing transform.
        jaynesize_transform!(ir)
    end
    ir
end

# Base fixes.
(ctx::UpdateContext)(::typeof(collect), b::Base.Generator) = collect(b)
(ctx::UpdateContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = sx(c, flatten(args)...)

# ------------ includes ------------ #

include("dynamic/regenerate.jl")
include("factor/regenerate.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct RegenerateContext{T <: Trace, 
                                 L <: Target,
                                 P <: AddressMap} <: ExecutionContext
    prev::T
    tr::T
    target::L
    weight::Float64
    score::Float64
    discard::T
    visited::Visitor
    params::P
end
```

Inner constructors:

```julia
function RegenerateContext(tr::T, sel::Vector{Address}) where T <: Trace
    un_sel = targetion(sel)
    new{T, typeof(un_sel), EmptyAddressMap}(tr, Trace(), un_sel, 0.0, Trace(), Visitor(), AddressMap())
end
function RegenerateContext(tr::T, sel::L) where {T <: Trace, L <: Target}
    new{T, L, EmptyAddressMap}(tr, Trace(), sel, 0.0, Trace(), Visitor(), AddressMap())
end
```

Outer constructors:

```julia
Regenerate(tr::Trace, sel::Vector{Address}) = RegenerateContext(tr, sel)
Regenerate(tr::Trace, sel::Target) = RegenerateContext(tr, sel)
```

The `RegenerateContext` is used for MCMC algorithms, to propose new choices for addresses indicated by an `Target` in the `target` field.
""", RegenerateContext)

@doc(
"""
```julia
ret, cl = regenerate(sel::L, cs::DynamicCallSite, new_args...) where L <: Target
ret, cl = regenerate(sel::L, cs::DynamicCallSite) where L <: Target
```
`regenerate` is an API to the `RegenerateContext` execution context. `regenerate` requires that users provide an `Target`, an original call site, and possibly a set of new arguments to be used in the regeneration step. This context internally keeps track of the bookkeeping required to increment likelihood weights, as well as prune off parts of the trace which are invalid if a regenerated choice changes the shape of the trace (e.g. control flow), and returns a new return value `ret` as well as the modified call site `cl`.
""", regenerate)
