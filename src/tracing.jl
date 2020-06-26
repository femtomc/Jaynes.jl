# Generate.
function trace(ctx::TraceCtx{M},
               fn::Function, 
               args::Tuple) where M <: GenerateMeta
    ret = Cassette.overdub(ctx, fn, args...)
    return CallSite(ctx.metadata.tr, fn, args, ret)
end

function trace(fn::Function, 
               args::Tuple)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace()), pass = ignore_pass))
    return trace(ctx, fn, args)
end

function trace(fn::Function, 
               args::Tuple,
               obs::Vector{Tuple{K, T}}) where {K <: Union{Symbol, Pair}, T}
    ctx = disablehooks(TraceCtx(metadata = ConstrainedGenerateMeta(Trace(), obs), pass = ignore_pass))
    return trace(ctx, fn, args)
end

function trace(fn::Function, 
               args::Tuple,
               obs::ConstrainedSelection)
    ctx = disablehooks(TraceCtx(metadata = ConstrainedGenerateMeta(Trace(), obs), pass = ignore_pass))
    return trace(ctx, fn, args)
end

# Regenerate.
function visited_walk!(discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, par_addr => addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, tr::T, vs::VisitedSelection) where T <: Trace
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(par_addr, discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
        
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(par_addr => addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
end

# Toplevel.
function visited_walk!(tr::T, vs::VisitedSelection) where T <: Trace
    discard = ConstrainedHierarchicalSelection()
    d_score = 0.0
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
       
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
    tr.score -= d_score
    return discard
end

function trace(ctx::TraceCtx{M}, 
               fn::Function, 
               args::Tuple) where M <: RegenerateMeta
    ret = Cassette.overdub(ctx, fn, args...)
    discard = visited_walk!(ctx.metadata.tr, ctx.metadata.visited)
    return CallSite(ctx.metadata.tr, fn, args, ret), discard
end

# Update.
function trace(ctx::TraceCtx{M},
               fn::Function,
               args::Tuple) where M <: UpdateMeta
    ret = Cassette.overdub(ctx, fn, args...)
    # TODO: fix.
#    !foldl((x, y) -> x && y, map(ctx.metadata.constraints_visited) do k
#               k in keys(ctx.metadata.constraints)
#           end) && error("UpdateError: tracing did not visit all addresses in constraints.")
#
    discard = visited_walk!(ctx.metadata.tr, ctx.metadata.visited)
    return CallSite(ctx.metadata.tr, fn, args, ret), discard
end

# Score.
function trace(ctx::TraceCtx{M},
               fn::Function,
               args::Tuple) where M <: ScoreMeta
    ret = Cassette.overdub(ctx, fn, args...)
    return ctx.metadata.score
end
