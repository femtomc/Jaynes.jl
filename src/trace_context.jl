Cassette.@context TraceCtx

function Cassette.overdub(ctx::TraceCtx, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                         args) where T <: Union{Symbol, Pair}
    dist = dist(args...)
    sample = rand(dist)
    !isempty(ctx.metadata.stack) && begin
        addr = stack[end] => addr
    end
    addr in keys(ctx.metadata.chm) && error("AddressError: each address within a call must be unique. Found duplicate $(addr).")
    addr in keys(ctx.metadata.obs) && begin
        sample = ctx.metadata.obs[addr]
    end
    score = logpdf(dist, sample)
    ctx.metadata.chm[addr] = ChoiceOrCall(sample, score)
    ctx.metadata.score += score
    return sample
end

function Cassette.overdub(ctx::TraceCtx,
                          c::typeof(rand),
                          addr::T,
                          call::Function,
                          args) where T <: Union{Symbol, Pair}
    push!(ctx.metadata, addr)
    !isempty(args) && begin
        res = recurse(ctx, call, args)
        return res
    end
    res = recurse(ctx, call)
    return res
end

function trace(fn::Function, args)
    ctx = TraceCtx(metadata = Trace())
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.func = fn
    ctx.metadata.args = args
    ctx.metadata.retval = res
    return ctx.metadata
end
