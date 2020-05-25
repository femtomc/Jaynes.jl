function regenerate(tr::Trace, args::Tuple, selection::Vector{Address})
    ctx = TraceCtx(metadata = RegenerateMeta(tr, selection))
    func = tr.func
    ret = Cassette.overdub(ctx, func, args...)
    tr.retval = ret
    tr.args = args
    return ctx.metadata.tr, ctx.metadata.tr.score
end

function update(tr::Trace, args::Tuple, constraints::Dict{Address, T}) where T
    ctx = TraceCtx(metadata = UpdateMeta(tr, constraints))
    func = tr.func
    ret = Cassette.overdub(ctx, func, args...)
    tr.retval = ret
    tr.args = args
    score = ctx.metadata.tr.score
    !isempty(ctx.metadata.constraints) && begin
        error("UpdateError: tracing did not visit all addresses in constraints.")
    end
    return ctx.metadata.tr, ctx.metadata.tr.score
end

# Convenience.
function trace(fn::Function)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata.tr, ctx.metadata.tr.score
end

function trace(fn::Function, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata.tr, ctx.metadata.tr.score
end

function trace(fn::Function, args::Tuple)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata.tr, ctx.metadata.tr.score
end

function trace(fn::Function, args::Tuple, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata.tr, ctx.metadata.tr.score
end
