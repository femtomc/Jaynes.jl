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
function (sx::RegenerateContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...)
    flt = flatten(args)
    addr, rest = flt[1], flt[2 : end]
    ret, cl = regenerate(rest...)
    add_call!(sx, addr, cl)
    ret
end

function (sx::RegenerateContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        δ = Δ(i, NoChange())
        sx(generator.f, tupletype(δ), δ)
    end
end

function regenerate(e::E, args...) where E <: ExecutionContext
    ctx = Regenerate(Trace(), Empty())
    ret = ctx(e, args...)
    return ret, DynamicCallSite(ctx.tr, ctx.score, e, args, ret)
end

# ------------ Choice sites ------------ #

@inline function (ctx::RegenerateContext)(call::typeof(trace), 
                                          addr::T, 
                                          d::Distribution{K}) where {T <: Address, K}
    visit!(ctx, addr)
    in_prev_chm = has_value(get_trace(ctx.prev), addr)
    in_sel = haskey(ctx.target, addr)

    if in_prev_chm
        prev = get_sub(get_trace(ctx.prev), addr)
    end

    if in_sel && in_prev_chm
        ret = rand(d)
        set_sub!(ctx.discard, addr, prev)
    elseif in_prev_chm
        ret = prev.val
    else
        ret = rand(d)
    end

    score = logpdf(d, ret)
    in_prev_chm && increment!(ctx, score - get_score(prev))
    add_choice!(ctx, addr, score, ret)
    return ret
end

# ------------ Learnable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return getindex(ctx.params, addr)
    error("(learnable): parameter not provided at address $addr.")
end

# ------------ Fillable ------------ #

@inline function (ctx::RegenerateContext)(fn::typeof(fillable), addr::Address)
    haskey(ctx.target, addr) && return getindex(ctx.target, addr)
    error("(fillable): parameter not provided at address $addr.")
end

# ------------ Black box call sites ------------ #

@inline function (ctx::RegenerateContext)(c::typeof(trace),
                                          addr::T,
                                          call::Function,
                                          args...) where T <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(get_trace(ctx.prev), addr)
        prev_call = get_prev(ctx, addr)
        ret, cl, w, retdiff, d = regenerate(ss, ps, prev_call, UnknownChange(), args...)
    else
        ret, cl, w = generate(ss, ps, call, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

@inline function (ctx::RegenerateContext)(c::typeof(trace),
                                          addr::T,
                                          call::G,
                                          args...) where {G <: GenerativeFunction, T <: Address}
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ss = get_sub(ctx.target, addr)
    if has_sub(get_trace(ctx.prev), addr)
        prev_call = get_prev(ctx, addr)
        ret, cl, w, retdiff, d = regenerate(ss, ps, prev_call, UnknownChange(), args...)
    else
        ret, cl, w = generate(ss, ps, call.fn, args...)
    end
    add_call!(ctx, addr, cl)
    increment!(ctx, w)
    return ret
end

# ------------ Utilities ------------ #

function regenerate_projection_walk(tr::DynamicTrace,
                                    visited::Visitor)
    weight = 0.0
    for (k, v) in shallow_iterator(tr)
        if !(k in visited)
            weight += projection(v, SelectAll())[1]
        end
    end
    weight
end

function regenerate_discard_walk!(d::DynamicDiscard,
                                  visited::Visitor,
                                  prev::DynamicTrace)
    for (k, v) in shallow_iterator(prev)
        if !(k in visited)
            ss = get_sub(visited, k)
            if isempty(ss)
                set_sub!(d, k, v)
            else
                sd = get_sub(d, k)
                sd = isempty(sd) ? DynamicMap{Value}() : sd
                discard_walk!(sd, ss, v)
                set_sub!(d, k, sd)
            end
        end
    end
end

# ------------ Convenience ------------ #

function regenerate(ctx::RegenerateContext, cs::DynamicCallSite, args::NTuple{N, Diffed}) where N
    ret = ctx(cs.fn, tupletype(args...), args...)
    adj_w = regenerate_projection_walk(ctx.tr, ctx.visited)
    regenerate_discard_walk!(ctx.discard, ctx.visited, get_trace(cs))
    return ret, DynamicCallSite(ctx.tr, ctx.score - adj_w, cs.fn, map(a -> unwrap(a), args), ret), ctx.weight, UnknownChange(), ctx.discard
end

function regenerate(sel::L, cs::DynamicCallSite) where L <: AddressMap
    ctx = Regenerate(sel, Empty(), cs, DynamicTrace(), DynamicDiscard())
    return regenerate(ctx, cs, map(cs.args) do a
                          Δ(a, NoChange())
                      end)
end

function regenerate(sel::L, cs::DynamicCallSite, args::Diffed...) where {L <: AddressMap, N}
    ctx = Regenerate(sel, Empty(), cs, DynamicTrace(), DynamicDiscard())
    return regenerate(ctx, cs, args)
end

function regenerate(sel::L, ps::P, cs::DynamicCallSite) where {P <: AddressMap, L <: AddressMap}
    ctx = Regenerate(sel, ps, cs, DynamicTrace(), DynamicDiscard())
    return regenerate(ctx, cs, map(cs.args) do a
                          Δ(a, NoChange())
                      end)
end

function regenerate(sel::L, ps::P, cs::DynamicCallSite, args::Diffed...) where {L <: AddressMap, P <: AddressMap, N}
    ctx = Regenerate(sel, ps, cs, DynamicTrace(), DynamicDiscard())
    return regenerate(ctx, cs, args)
end

function regenerate(sel::L, ps::P, cs::DynamicCallSite, args...) where {L <: AddressMap, P <: AddressMap, N}
    ctx = Regenerate(sel, ps, cs, DynamicTrace(), DynamicDiscard())
    args = map(args) do a
        a isa Diffed ? a : Δ(a, NoChange())
    end
    return regenerate(ctx, cs, args)
end

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
