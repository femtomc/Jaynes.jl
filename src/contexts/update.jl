# TODO: specialize to different call sites.
mutable struct UpdateContext{C <: CallSite, 
                             T <: Trace,
                             K <: ConstrainedSelection, 
                             P <: Parameters, 
                             D <: Diff} <: ExecutionContext
    prev::C
    tr::T
    select::K
    weight::Float64
    score::Float64
    discard::HierarchicalTrace
    visited::Visitor
    params::P
    argdiff::D
    # Re-write with dispatch for specialized vs. black box.
    UpdateContext(cl::C, select::K, argdiffs::D) where {C <: CallSite, K <: ConstrainedSelection, D <: Diff} = new{C, typeof(cl.trace), K, NoParameters, D}(cl, typeof(cl.trace)(), select, 0.0, 0.0, Trace(), Visitor(), Parameters(), argdiffs)
end

# Update has a special dynamo.
@dynamo function (mx::UpdateContext)(a...)
    ir = IR(a...)
    ir == nothing && return
    recur!(ir)
    return ir
end

# ------------ Convenience ------------ #

function update(ctx::UpdateContext, bbcs::HierarchicalCallSite, args...) where D <: Diff
    ret = ctx(bbcs.fn, args...)
    return ret, HierarchicalCallSite(ctx.tr, ctx.score, bbcs.fn, args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, bbcs::HierarchicalCallSite) where L <: ConstrainedSelection
    argdiffs = NoChange()
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, bbcs.args...)
end

function update(sel::L, bbcs::HierarchicalCallSite, argdiffs::D, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    ctx = UpdateContext(bbcs, sel, argdiffs)
    return update(ctx, bbcs, new_args...)
end

function update(sel::L, vcs::VectorizedSite{typeof(plate)}, argdiffs::D, new_args...) where {L <: ConstrainedSelection, D <: Diff}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, argdiffs)
    ret = ctx(plate, addr, vcs.kernel, new_args...)
    return ret, VectorizedSite{typeof(plate)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedSite{typeof(plate)}) where {L <: ConstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, argdiffs)
    ret = ctx(plate, addr, vcs.kernel, vcs.args)
    return ret, VectorizedSite{typeof(plate)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

# ------------ includes ------------ #

include("generic/update.jl")
include("plate/update.jl")
include("markov/update.jl")
