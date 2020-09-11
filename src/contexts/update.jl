# TODO: specialize to different call sites.
mutable struct UpdateContext{C <: CallSite, 
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

@inline function record_cached!(ctx::UpdateContext, addr)
    visit!(ctx, addr)
    sub = get_sub(ctx.prev, addr)
    sc = get_score(sub)
    ctx.score += get_score(sub)
    set_sub!(ctx.tr, addr, sub)
    get_value(sub)
end

function Update(select::K, ps::P, cl::C, tr, discard) where {K <: AddressMap, P <: AddressMap, C <: CallSite}
    UpdateContext(cl, 
                  tr,
                  select, 
                  0.0, 
                  0.0, 
                  discard,
                  Visitor(), 
                  ps)
end

# This uses a "sneaky invoke" hack to allow passage of diffs into user-defined functions whose argtypes do not allow it.
@dynamo function (mx::UpdateContext{C, T, K})(f, ::Type{S}, args...) where {S <: Tuple, C, T, K}

    # Check for primitive.
    ir = IR(f, S.parameters...)
    ir == nothing && return

    # Equivalent to static DSL optimizations.
    if !control_flow_check(ir) || K <: DynamicMap

        # Release IR normally.
        ir = recur(ir)
        argument!(ir, at = 2)
        ir = renumber(ir)
    else

        # Diff inference.
        args = map(args) do a
            create_flip_diff(a)
        end
        tr = _propagate(f, S.parameters, args)

        # Get choicemap keys.
        ks = get_address_schema(K)

        # Pruning transform.
        argument!(tr, at = 2)
        ir = pipeline(ir.meta, tr, ks)
    end
    display(ir)
    ir
end

# Base fixes.
(ctx::UpdateContext)(::typeof(collect), b::Base.Generator) = collect(b)

# ------------ includes ------------ #

include("dynamic/update.jl")
include("plate/update.jl")
include("markov/update.jl")
include("factor/update.jl")

# ------------ Documentation ------------ #

@doc(
"""
```julia
mutable struct UpdateContext{C <: CallSite, 
                             T <: AddressMap,
                             K <: AddressMap, 
                             P <: AddressMap, 
                             D <: Diff} <: ExecutionContext
    prev::C
    tr::T
    select::K
    weight::Float64
    score::Float64
    discard::DynamicAddressMap
    visited::Visitor
    params::P
    argdiffs::D
end
```

Inner constructor:

```julia
UpdateContext(cl::C, select::K, argdiffs::D) where {C <: CallSite, K <: AddressMap, D <: Diff} = new{C, typeof(cl.trace), K, EmptyAddressMap, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, AddressMap(), Visitor(), AddressMap(), argdiffs)
UpdateContext(cl::C, select::K, ps::P, argdiffs::D) where {C <: CallSite, K <: AddressMap, P <: AddressMap, D <: Diff} = new{C, typeof(cl.trace), K, EmptyAddressMap, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, AddressMap(), Visitor(), ps, argdiffs)
```

`UpdateContext` is an execution context used for updating the value of random choices in an existing recorded call site. This context will perform corrective updates to the likehood weights and scores so that this operation produces the correct weights and scores for the original model program constrained with the `select` selection in the `UpdateContext`.
""", UpdateContext)

@doc(
"""
```julia
ret, cl, w, retdiff, d = update(ctx::UpdateContext, cs::DynamicCallSite, args...) where D <: Diff
ret, cl, w, retdiff, d = update(sel::L, cs::DynamicCallSite) where L <: AddressMap
ret, cl, w, retdiff, d = update(sel::L, cs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(plate)}) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, d::NoChange, len::Int) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, D <: Diff}
```

`update` provides an API to the `UpdateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, the updated `RecordSite` instance `cl` or `v_cl`, the updated weight `w`, a `Diff` instance for the return value `retdiff`, and a structure which contains any changed (i.e. discarded) record sites `d`.
""", update)
