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
function visited_walk!(discard, d_score::Float64, rs::RecordSite, addr)
    discard[addr] = rs
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, rs::RecordSite, addr)
    discard[par_addr => addr] = rs
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
    discard = Dict{Union{Symbol, Pair}, RecordSite}() 
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
    ctx.metadata.fn = fn
    ctx.metadata.args = args
    ctx.metadata.ret = ret
    !foldl((x, y) -> x && y, map(ctx.metadata.constraints_visited) do k
               k in keys(ctx.metadata.constraints)
           end) && error("UpdateError: tracing did not visit all addresses in constraints.")

    # Discard.
    discard = typeof(ctx.metadata.tr.chm)()
    discard_score = 0.0
    for (k, v) in ctx.metadata.tr.chm
        !(k in ctx.metadata.visited) && begin
            discard_score += ctx.metadata.tr.chm[k].score
            discard[k] = v
            delete!(ctx.metadata.tr.chm, k)
        end
    end

    ctx.metadata.tr.score -= discard_score
    return ctx, ctx.metadata.tr, ctx.metadata.tr.score, discard
end

# Inference compilation.
function trace(ctx::TraceCtx{M}, 
               constraints::Dict{Address, T}) where {T, M <: InferenceCompilationMeta}
    !(length(constraints) == 1 && ctx.metadata.target in keys(constraints)) && begin
        error("InferenceCompilationError: constraints must contain the target address and only the target address.")
    end
    ctx.metadata.tr = Trace()
    ctx.metadata.constraints = constraints
    ret = Cassette.overdub(ctx, ctx.metadata.func, ctx.metadata.args...)
    ctx.metadata.ret = ret
    Flux.reset!(ctx.metadata.compiler.spine)
    return ctx, ctx.metadata.tr, ctx.metadata.tr.score
end

