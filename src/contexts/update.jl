# TODO: specialize to different call sites.
mutable struct UpdateContext{C <: CallSite, 
                             T <: AddressMap,
                             K <: AddressMap, 
                             D <: AddressMap,
                             P <: AddressMap, 
                             Ag <: Diff} <: ExecutionContext
    prev::C
    tr::T
    schema::K
    weight::Float64
    score::Float64
    discard::D
    visited::Visitor
    params::P
    argdiffs::Ag
end
function Update(select::K, cl::C) where {K <: AddressMap, C <: CallSite}
    UpdateContext(cl, 
                  typeof(cl.trace)(), 
                  select, 
                  0.0, 
                  0.0, 
                  DynamicDiscard(), 
                  Visitor(), 
                  Empty(), 
                  NoChange())
end
function Update(select::K, cl::C, argdiffs::Ag) where {K <: AddressMap, C <: CallSite, Ag <: Diff}
    UpdateContext(cl, 
                  typeof(cl.trace)(), 
                  select, 
                  0.0, 
                  0.0, 
                  DynamicDiscard(), 
                  Visitor(), 
                  Empty(), 
                  argdiffs)
end

# Update has a special dynamo.
@dynamo function (mx::UpdateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, bbcs::DynamicCallSite, args...) where D <: Diff
    ret = ctx(bbcs.fn, args...)
    visited = ctx.visited
    #adj_w = adjust_to_intersection(get_trace(bbcs), visited)
    adj_w = 0.0
    return ret, DynamicCallSite(ctx.tr, ctx.score - adj_w, bbcs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, bbcs::DynamicCallSite) where L <: AddressMap
    argdiffs = NoChange()
    ctx = Update(sel, bbcs, argdiffs)
    return update(ctx, bbcs, bbcs.args...)
end

function update(sel::L, ps::P, bbcs::DynamicCallSite) where {L <: AddressMap, P <: AddressMap}
    argdiffs = NoChange()
    ctx = UpdateContext(bbcs, sel, ps, argdiffs)
    return update(ctx, bbcs, bbcs.args...)
end

function update(bbcs::DynamicCallSite, argdiffs::D, new_args...) where D <: Diff
    sel = selection()
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(ps::P, bbcs::DynamicCallSite, argdiffs::D, new_args...) where {P <: AddressMap, D <: Diff}
    sel = selection()
    ctx = UpdateContext(bbcs, sel, ps, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, bbcs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, D <: Diff}
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, ps::P, bbcs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, P <: AddressMap, D <: Diff}
    ctx = UpdateContext(bbcs, sel, ps, argdiffs)
    return update(ctx, bbcs, new_args...)
end

# TODO: disallowed for now.
#function update(sel::L, vcs::VectorizedCallSite{typeof(plate)}, argdiffs::D, new_args...) where {L <: AddressMap, D <: Diff}
#    addr = gensym()
#    v_sel = selection(addr => sel)
#    ctx = UpdateContext(vcs, v_sel, argdiffs)
#    ret = ctx(plate, addr, vcs.fn, new_args...)
#    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
#end

function update(sel::L, vcs::VectorizedCallSite{typeof(plate)}) where L <: AddressMap
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, argdiffs)
    ret = ctx(plate, vcs.fn, vcs.args)
    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorizedCallSite{typeof(plate)}) where {L <: AddressMap, P <: AddressMap}
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, ps, argdiffs)
    ret = ctx(plate, vcs.fn, vcs.args)
    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedCallSite{typeof(markov)}) where L <: AddressMap
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, argdiffs)
    ret = ctx(markov, vcs.fn, vcs.args[1], vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorizedCallSite{typeof(markov)}) where {L <: AddressMap, P <: AddressMap}
    argdiffs = NoChange()
    ctx = UpdateContext(vcs, sel, ps, argdiffs)
    ret = ctx(markov, vcs.fn, vcs.args[1], vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, D <: Diff}
    ctx = UpdateContext(vcs, sel, NoChange())
    ret = ctx(markov, vcs.fn, len, vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, ps::P, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, P <: AddressMap, D <: Diff}
    ctx = UpdateContext(vcs, sel, ps, NoChange())
    ret = ctx(markov, vcs.fn, len, vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.fn, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

# ------------ includes ------------ #

include("dynamic/update.jl")
#include("plate/update.jl")
#include("markov/update.jl")
#include("factor/update.jl")

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
ret, cl, w, retdiff, d = update(ctx::UpdateContext, bbcs::DynamicCallSite, args...) where D <: Diff
ret, cl, w, retdiff, d = update(sel::L, bbcs::DynamicCallSite) where L <: AddressMap
ret, cl, w, retdiff, d = update(sel::L, bbcs::DynamicCallSite, argdiffs::D, new_args...) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(plate)}) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, d::NoChange, len::Int) where {L <: AddressMap, D <: Diff}
ret, v_cl, w, retdiff, d = update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: AddressMap, D <: Diff}
```

`update` provides an API to the `UpdateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, the updated `RecordSite` instance `cl` or `v_cl`, the updated weight `w`, a `Diff` instance for the return value `retdiff`, and a structure which contains any changed (i.e. discarded) record sites `d`.
""", update)
