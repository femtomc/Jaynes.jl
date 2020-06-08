Cassette.@context TraceCtx

# ------------------- META ----------------- #

# Structured metadata. This acts as dispatch on overdub - increases the efficiency of the system and forms the core set of interfaces for inference algorithms to use.
# For each inference interface, there are typically only a few constant pieces of Meta - these pieces tend to keep constant allocations out of sampling loops.
abstract type Meta end

mutable struct UnconstrainedGenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    visited::Vector{Address}
    args::Tuple
    fn::Function
    ret::Any
    UnconstrainedGenerateMeta(tr::Trace) = new(tr, Address[], Address[])
end
Generate(tr::Trace) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGenerateMeta(tr)))

mutable struct GenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    visited::Vector{Address}
    constraints::ConstrainedSelection
    args::Tuple
    fn::Function
    ret::Any
    GenerateMeta(tr::Trace, constraints::ConstrainedSelection) where T = new(tr, Address[], Address[], constraints)
end
Generate(tr::Trace, constraints) = disablehooks(TraceCtx(metadata = GenerateMeta(tr, constraints)))
Generate(pass, tr::Trace, constraints) = disablehooks(TraceCtx(pass = pass, metadata = GenerateMeta(tr, constraints)))

mutable struct ProposalMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    visited::Vector{Address}
    args::Tuple
    fn::Function
    ret::Any
    ProposalMeta(tr::Trace) = new(tr, Address[], Address[])
end
Propose(tr::Trace) = disablehooks(TraceCtx(metadata = ProposalMeta(tr)))
Propose(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = ProposalMeta(tr)))

mutable struct UpdateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    visited::Vector{Address}
    constraints_visited::Vector{Address}
    constraints::ConstrainedSelection
    args::Tuple
    fn::Function
    ret::Any
    UpdateMeta(tr::Trace, constraints::ConstrainedSelection) = new(tr, Address[], Address[], Address[], constraints)
end
Update(tr::Trace, constraints) where T = disablehooks(TraceCtx(metadata = UpdateMeta(tr, constraints)))
Update(pass, tr::Trace, constraints) where T = disablehooks(TraceCtx(pass = pass, metadata = UpdateMeta(tr, constraints)))

mutable struct RegenerateMeta <: Meta
    tr::Trace
    stack::Vector{Address}
    visited::Vector{Address}
    selection::UnconstrainedSelection
    args::Tuple
    fn::Function
    ret::Any
    RegenerateMeta(tr::Trace, sel::Vector{Address}) = new(tr, Address[], Address[], selection(sel))
end
Regenerate(tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(metadata = RegenerateMeta(tr, sel)))
Regenerate(pass, tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(pass = pass, metadata = RegenerateMeta(tr, sel)))

mutable struct ScoreMeta <: Meta
    tr::Trace
    score::Float64
    stack::Vector{Address}
    visited::Vector{Address}
    args::Tuple
    fn::Function
    ret::Any
    Score(tr::Trace) = new(tr, 0.0, Address[], Address[])
end
Score(tr::Trace) = disablehooks(TraceCtx(metadata = Score(tr)))
Score(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = Score(tr)))

# Required to track nested calls in overdubbing.
import Base: push!, pop!

function push!(trm::T, call::Address) where T <: Meta
    push!(trm.stack, call)
end

function pop!(trm::T) where T <: Meta
    pop!(trm.stack)
end

function reset_keep_constraints!(ctx::TraceCtx{M}) where M <: Meta
    ctx.metadata.tr = Trace()
    ctx.metadata.visited = Address[]
end

# --------------- OVERDUB -------------------- #

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: UnconstrainedGenerateMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    push!(ctx.metadata.visited, addr)
    return sample
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: GenerateMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)

    # Constrained..
    if haskey(ctx.metadata.constraints, addr)
        sample = ctx.metadata.constraints[addr]
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        ctx.metadata.tr.score += score
        push!(ctx.metadata.visited, addr)
        return sample

        # Unconstrained.
    else
        sample = rand(d)
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        push!(ctx.metadata.visited, addr)
        return sample
    end
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: ProposalMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = Choice(sample, score)
    ctx.metadata.tr.score += score
    push!(ctx.metadata.visited, addr)
    return sample

end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: RegenerateMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.tr.chm[addr]
        prev_val = prev.val
        prev_score = prev.score
    end

    # Check if in selection in meta.
    selection = ctx.metadata.selection.addresses
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

    # Visited
    push!(ctx.metadata.visited, addr)
    ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: UpdateMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
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
    in_constraints = haskey(ctx.metadata.constraints, addr)

    # Ret.
    d = dist(args...)
    if in_constraints
        ret = ctx.metadata.constraints[addr]
        push!(ctx.metadata.constraints_visited, addr)
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

    # Visited.
    push!(ctx.metadata.visited, addr)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: ScoreMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr(=>, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Get val.
    val = ctx.metadata.tr.chm[addr].value
    d = dist(args...)
    ctx.metadata.tr.score += logpdf(d, val)

    # Visited.
    push!(ctx.metadata.visited, addr)
    return val
end

# ------------------ END OVERDUB ------------------- #

# This handles functions (not distributions) in rand calls. When we see a rand call with a function, we push the address for that rand call onto the stack, and then recurse into the function. This organizes the choice map in the correct hierarchical way.
@inline function Cassette.overdub(ctx::TraceCtx,
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function) where T <: Address
    push!(ctx.metadata, addr)
    ret = recurse(ctx, call)
    pop!(ctx.metadata)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx,
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args) where T <: Address
    push!(ctx.metadata, addr)
    ret = recurse(ctx, call, args...)
    pop!(ctx.metadata)
    return ret
end

# Fallback - encounter some Cassette issues if we don't force this override for some Core functions.
@inline Cassette.fallback(ctx::TraceCtx, c::Function, args) = c(args)
