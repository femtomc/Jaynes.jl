Cassette.@context TraceCtx

# Structured metadata. This acts as dispatch on overdub - increases the efficiency of the system.
abstract type Meta end

mutable struct UnconstrainedGenerateMeta <: Meta
    tr::Trace
    stack::Vector{Union{Symbol, Pair}}
    UnconstrainedGenerateMeta(tr::Trace) = new(tr, Symbol[])
end

mutable struct GenerateMeta{T} <: Meta
    tr::Trace
    constraints::T
    stack::Vector{Union{Symbol, Pair}}
    GenerateMeta(tr::Trace, constraints::T) where T = new{T}(tr, constraints, Symbol[])
end

mutable struct ProposalMeta <: Meta
    tr::Trace
    stack::Vector{Union{Symbol, Pair}}
    ProposalMeta(tr::Trace) = new(tr, Symbol[])
end

# Required to track nested calls in overdubbing.
import Base: push!, pop!

function push!(trm::T, call::Address) where T <: Meta
    push!(trm.stack, call)
end

function pop!(trm::T) where T <: Meta
    pop!(trm.stack)
end

function reset_keep_constraints!(trm::T) where T <: Meta
    trm.tr = Trace()
    trm.stack = Union{Symbol, Pair}[]
end

# --------------- OVERDUB -------------------- #

function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: UnconstrainedGenerateMeta, 
                                       T <: Union{Symbol, Pair}}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    haskey(ctx.metadata.tr.chm, addr) && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    dist = dist(args...)
    sample = rand(dist)
    score = logpdf(dist, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    return sample
end

function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: GenerateMeta, 
                                       T <: Union{Symbol, Pair}}
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
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        ctx.metadata.tr.score += score
        return sample

    # Unconstrained.
    else
        sample = rand(dist)
        score = logpdf(dist, sample)
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        return sample
    end
end

function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: ProposalMeta, 
                                       T <: Union{Symbol, Pair}}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    haskey(ctx.metadata.tr.chm, addr) && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    dist = dist(args...)
    sample = rand(dist)
    score = logpdf(dist, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    ctx.metadata.tr.score += score
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
        pop!(ctx.metadata)
        return res
    end
    res = recurse(ctx, call)
    pop!(ctx.metadata)
    return res
end

# Convenience.
function trace(fn::Function)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = ()
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, args::Tuple)
    ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end

function trace(fn::Function, args::Tuple, constraints::Dict{Address, T}) where T
    ctx = disablehooks(TraceCtx(metadata = GenerateMeta(Trace(), constraints)))
    res = Cassette.overdub(ctx, fn, args...)
    ctx.metadata.tr.func = fn
    ctx.metadata.tr.args = args
    ctx.metadata.tr.retval = res
    return ctx.metadata, ctx.metadata.tr
end
