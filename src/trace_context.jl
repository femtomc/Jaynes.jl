Cassette.@context TraceCtx

# ------------------- META ----------------- #

# Structured metadata. This acts as dispatch on overdub - increases the efficiency of the system and forms the core set of interfaces for inference algorithms to use.
abstract type Meta end

mutable struct UnconstrainedGenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    UnconstrainedGenerateMeta(tr::Trace) = new(tr, Address[])
end

mutable struct GenerateMeta{T} <: Meta
    tr::Trace
    stack::Vector{Address}
    constraints::T
    GenerateMeta(tr::Trace, constraints::T) where T = new{T}(tr, Address[], constraints)
end

mutable struct ProposalMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    ProposalMeta(tr::Trace) = new(tr, Address[])
end

mutable struct UpdateMeta{T} <: Meta
    tr::Trace
    stack::Vector{Address}
    constraints::T
    UpdateMeta(tr::Trace, constraints::T) where T = new{T}(tr, Address[], constraints)
end

mutable struct RegenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    selection::Vector{Address}
    RegenerateMeta(tr::Trace, sel::Vector{Address}) = new(tr, Address[], sel)
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

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
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

    d = dist(args...)

    # Constrained..
    if haskey(ctx.metadata.constraints, addr)
        sample = ctx.metadata.constraints[addr]
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        ctx.metadata.tr.score += score
        return sample

    # Unconstrained.
    else
        sample = rand(d)
        score = logpdf(d, sample)
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

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    ctx.metadata.tr.score += score
    return sample

end

function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: RegenerateMeta, 
                                       T <: Union{Symbol, Pair}}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.tr.chm[addr]
        prev_val = prev.val
        prev_score = prev.score
    end

    # Check if in selection (part of meta).
    selection = ctx.metadata.selection
    in_sel = addr in selection

    d = dist(args...)
    ret = rand(d)
    in_prev_chm && !in_sel && begin
        ret = prev_val
    end

    score = logpdf(d, ret)

    in_prev_chm && !in_sel && begin
        ctx.metadata.tr.score += score - prev_score
    end
    ctx.metadata.tr.chm[addr] = Choice(ret, score)
    ret
end

function Cassette.overdub(ctx::TraceCtx{M}, 
                          call::typeof(rand), 
                          addr::T, 
                          dist::Type,
                          args) where {N, 
                                       M <: UpdateMeta, 
                                       T <: Union{Symbol, Pair}}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.tr.chm[addr]
        prev_ret = prev.val
        prev_score = prev.score
    end
    
    # Check if in constraints.
    in_constraints = haskey(ctx.metadata.tr.constraints, addr)
    if in_constraints
        ret = ctx.metadata.tr.constraints[addr]
    end

    # Ret.
    d = dist(args...)
    if in_constraints
        ret = ctx.metadata.tr.constraints[addr]
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        ctx.metadata.tr.score += score - prev_score
    elseif in_constraints
        ctx.metadata.tr.score += score
    end
    ctx.metadata.tr.chm[addr] = Choice(ret, score)
    return ret
end

# This handles functions (not distributions) in rand calls. When we see a rand call with a function, we push the address for that rand call onto the stack, and then recurse into the function. This organizes the choice map in the correct hierarchical way.
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
