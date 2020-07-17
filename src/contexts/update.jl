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
    argdiffs::D
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

# TODO: disallowed for now.
#function update(sel::L, vcs::VectorizedCallSite{typeof(plate)}, argdiffs::D, new_args...) where {L <: ConstrainedSelection, D <: Diff}
#    addr = gensym()
#    v_sel = selection(addr => sel)
#    ctx = UpdateContext(vcs, v_sel, argdiffs)
#    ret = ctx(plate, addr, vcs.kernel, new_args...)
#    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
#end

function update(sel::L, vcs::VectorizedCallSite{typeof(plate)}) where {L <: ConstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, argdiffs)
    ret = ctx(plate, addr, vcs.kernel, vcs.args)
    return ret, VectorizedCallSite{typeof(plate)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedCallSite{typeof(markov)}) where {L <: ConstrainedSelection, D <: Diff}
    argdiffs = NoChange()
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, argdiffs)
    ret = ctx(markov, addr, vcs.kernel, vcs.args[1], vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, d::NoChange, len::Int) where {L <: ConstrainedSelection, D <: Diff}
    addr = gensym()
    v_sel = selection(addr => sel)
    ctx = UpdateContext(vcs, v_sel, d)
    ret = ctx(markov, addr, vcs.kernel, len, vcs.args[2]...)
    return ret, VectorizedCallSite{typeof(markov)}(ctx.tr, ctx.score, vcs.kernel, vcs.args, ret), ctx.weight, UndefinedChange(), ctx.discard
end

function update(sel::L, vcs::VectorizedCallSite{typeof(markov)}, len::Int) where {L <: ConstrainedSelection, D <: Diff}
    return update(sel, vcs, NoChange(), len)
end

# ------------ includes ------------ #

include("hierarchical/update.jl")
include("plate/update.jl")
include("markov/update.jl")
