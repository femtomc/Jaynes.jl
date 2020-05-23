Cassette.@context TraceCtx

function Cassette.overdub(ctx::TraceCtx, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                         args) where T <: Union{Symbol, Pair}
    # Check for support errors.
    !isempty(ctx.metadata.stack) && begin
        addr = ctx.metadata.stack[end] => addr
    end
    addr in keys(ctx.metadata.chm) && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")
        
    dist = dist(args...)
    # Constrained..
    if addr in keys(ctx.metadata.obs)
        sample = ctx.metadata.obs[addr]
        score = logpdf(dist, sample)
        ctx.metadata.chm[addr] = ChoiceOrCall(sample, score)
        ctx.metadata.score += score
        return sample

    # Unconstrained.
    else
        sample = rand(dist)
        score = logpdf(dist, sample)
        ctx.metadata.chm[addr] = ChoiceOrCall(sample, score)
        return sample
    end
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

function trace(fn::Function, args::Tuple)
    ctx = TraceCtx(metadata = Trace())
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.func = fn
    ctx.metadata.args = args
    ctx.metadata.retval = res
    return ctx.metadata
end
