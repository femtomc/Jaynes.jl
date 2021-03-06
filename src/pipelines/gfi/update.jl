# ------------ Choice sites ------------ #

@inline function (ctx::UpdateContext)(call::typeof(trace), 
                                      addr::T, 
                                      d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)

    # Check if in previous trace's choice map.
    in_prev_chm = has_value(ctx.prev, addr)
    in_prev_chm && begin
        prev = get_sub(ctx.prev, addr)
        prev_ret = get_value(prev)
        prev_score = get_score(prev)
    end

    # Check if in target.
    in_target = has_value(ctx.target, addr)

    # Ret.
    if in_target
        ret = getindex(ctx.target, addr)
        in_prev_chm && begin
            set_sub!(ctx.discard, addr, prev)
        end
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        increment!(ctx, score - prev_score)
    elseif in_target
        increment!(ctx, score)
    end
    add_choice!(ctx, addr, score, ret)

    return ret
end

# ------------ Learnable ------------ #

@inline function (ctx::UpdateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::UpdateContext)(c::typeof(trace),
                                      addr::T,
                                      call::Function,
                                      args...) where {T <: Address, D <: Diff}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(ctx.prev, addr)
        prev = get_prev(ctx, addr)
        ret, cl, w, rd, d = update(ss, ps, prev, args...)
    else
        ret, cl, w = generate(ss, ps, call, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

@inline function (ctx::UpdateContext)(c::typeof(trace),
                                      addr::T,
                                      call::G,
                                      args...) where {G <: GenerativeFunction,
                                                      T <: Address}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(ctx.prev, addr)
        prev = get_prev(ctx, addr)
        ret, cl, w, rd, d = update(ss, ps, prev, args...)
    else
        ret, cl, w = generate(ss, ps, call.fn, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Utilities ------------ #

# TODO: re-write with new projection.
function update_projection_walk(tr::DynamicTrace,
                                visited::Visitor)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        k in visited && continue
        weight += projection(v, SelectAll())[1]
    end
    weight
end

function update_discard_walk!(d::DynamicDiscard,
                              visited::Visitor,
                              prev::DynamicTrace)
    for (k, v) in shallow_iterator(prev)
        if !(k in visited)
            ss = get_sub(visited, k)
            if isempty(ss)
                set_sub!(d, k, v)
            else
                sd = get_sub(d, k)
                sd = isempty(sd) ? Empty() : sd
                update_discard_walk!(sd, ss, v)
                set_sub!(d, k, sd)
            end
        end
    end
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, cs::DynamicCallSite, args::NTuple{N, Diffed}) where N
    cs.fn isa JFunction ? func = get_fn(cs.fn) : func = cs.fn
    ret = ctx(func, tupletype(args...), args...)
    adj_w = update_projection_walk(ctx.tr, ctx.visited)
    update_discard_walk!(ctx.discard, ctx.visited, unwrap(get_trace(cs)))
    return ret, DynamicCallSite(ctx.tr, ctx.score - adj_w, cs.fn, map(a -> unwrap(a), args), ret), ctx.weight, UnknownChange(), ctx.discard
end

# Explicitly parametrize the compilation pipeline.
function update(opt::J, sel::L, ps::P, cs::DynamicCallSite, args::Diffed...) where {J <: CompilationOptions, L <: AddressMap, P <: AddressMap, N}
    ctx = Update(opt, sel, ps, cs, DynamicTrace(), DynamicDiscard())
    return update(ctx, cs, args)
end

# Explicitly parametrize the compilation pipeline.
function update(opt::J, sel::L, ps::P, cs::DynamicCallSite, args...) where {J <: CompilationOptions, L <: AddressMap, P <: AddressMap, N}
    ctx = Update(opt, sel, ps, cs, DynamicTrace(), DynamicDiscard())
    args = map(args) do a
        a isa Diffed ? a : Δ(a, NoChange())
    end
    return update(ctx, cs, args)
end

function update(sel::L, ps::P, cs::DynamicCallSite, args...) where {L <: AddressMap, P <: AddressMap, N}
    ctx = Update(DefaultPipeline(), sel, ps, cs, DynamicTrace(), DynamicDiscard())
    args = map(args) do a
        a isa Diffed ? a : Δ(a, NoChange())
    end
    return update(ctx, cs, args)
end

function update(sel::L, cs::DynamicCallSite, args...) where L <: AddressMap
    ctx = Update(DefaultPipeline(), sel, Empty(), cs, DynamicTrace(), DynamicDiscard())
    if isempty(args)
        args = get_args(cs)
    end
    args = map(args) do a
        a isa Diffed ? a : Δ(a, NoChange())
    end
    return update(ctx, cs, args)
end

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
```

`update` provides an API to the `UpdateContext` execution context. You can use this function on any of the matching signatures above - it will return the return value `ret`, the updated `RecordSite` instance `cl` or `v_cl`, the updated weight `w`, a `Diff` instance for the return value `retdiff`, and a structure which contains any changed (i.e. discarded) record sites `d`.
""", update)
