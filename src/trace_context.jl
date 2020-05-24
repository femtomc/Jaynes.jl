Cassette.@context TraceCtx

mutable struct TraceMeta{T}
    tr::Trace
    constraints::T
    stack::Vector{Symbol}
    TraceMeta(tr::Trace, constraints::T) where T = new{T}(tr, constraints, Symbol[])
end

# Required to track nested calls in IR.
import Base: push!, pop!

function push!(tr::Trace, call::Symbol)
    push!(tr.stack, call)
end

function pop!(tr::Trace)
    pop!(tr.stack)
end

function reset_keep_constraints!(trm::TraceMeta)
    trm.tr = Trace()
    trm.stack = Symbol[]
end

function Cassette.overdub(ctx::TraceCtx, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                         args) where T <: Union{Symbol, Pair}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end
    
    # Check for support errors.
    haskey(ctx.metadata.tr.chm, addr) && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")
        
    dist = dist(args...)

    # Constrained..
    if haskey(ctx.metadata.constraints, addr)
        sample = ctx.metadata.constraints[addr]
        score = logpdf(dist, sample)
        ctx.metadata.tr.chm[addr] = ChoiceOrCall(sample, score)
        ctx.metadata.tr.score += score
        return sample

    # Unconstrained.
    else
        sample = rand(dist)
        score = logpdf(dist, sample)
        ctx.metadata.tr.chm[addr] = ChoiceOrCall(sample, score)
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
        pop!(ctx.metadata)
        return res
    end
    res = recurse(ctx, call)
    pop!(ctx.metadata)
    return res
end

# Convenience.
function trace(fn::Function)
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), nothing)))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, args::Tuple)
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), nothing)))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, args::Tuple, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = TraceMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end
