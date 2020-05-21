Cassette.@context TraceCtx

function Cassette.overdub(ctx::TraceCtx, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                         args) where T <: Union{Symbol, Pair}
    dist = dist(args...)
    sample = rand(dist)
    stack = filter(unique(ctx.metadata.stack)) do k
        typeof(k) != typeof(rand)
    end
    obs = ctx.metadata.obs
    chm = ctx.metadata.chm
    !isempty(stack) && begin
        addr = stack[end] => addr
    end
    addr in keys(chm) && error("AddressError: each address within a call must be unique. Found duplicate $(addr).")
    addr in keys(obs) && begin
        sample = obs[addr]
    end
    lpdf = logpdf(dist, sample)
    ctx.metadata.chm[addr] = ChoiceOrCall(sample, lpdf, dist)
    ctx.metadata.score += lpdf
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
    return res, ctx.metadata
end
