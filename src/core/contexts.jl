Cassette.@context TraceCtx

# ------------------- META ----------------- #

abstract type Meta end
abstract type Effect end
abstract type None <: Effect end

mutable struct UnconstrainedGenerateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    UnconstrainedGenerateMeta(tr::T) where T <: Trace = new{T}(tr, Address[])
end
Generate(tr::Trace) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(tr::Trace, constraints::EmptySelection) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace, constraints::EmptySelection) = disablehooks(TraceCtx(pass = pass, metadata = GenerateMeta(tr, constraints)))

mutable struct GenerateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    constraints::ConstrainedHierarchicalSelection
    GenerateMeta(tr::T, constraints::ConstrainedHierarchicalSelection) where T <: Trace = new{None}(tr, Address[], Union{Symbol,Pair}[], constraints)
end
Generate(tr::Trace, constraints::ConstrainedHierarchicalSelection) = disablehooks(TraceCtx(metadata = GenerateMeta(tr, constraints)))
Generate(pass, tr::Trace, constraints) = disablehooks(TraceCtx(pass = pass, metadata = GenerateMeta(tr, constraints)))

mutable struct ProposalMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    ProposalMeta(tr::T) where T <: Trace = new{T}(tr, Address[])
end
Propose(tr::Trace) = disablehooks(TraceCtx(metadata = ProposalMeta(tr)))
Propose(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = ProposalMeta(tr)))

mutable struct UpdateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    constraints_visited::Vector{Address}
    constraints::ConstrainedHierarchicalSelection
    UpdateMeta(tr::T, constraints::ConstrainedHierarchicalSelection) where T <: Trace = new{T}(tr, Address[], Union{Symbol, Pair}[], constraints)
end
Update(tr::Trace, constraints) where T = disablehooks(TraceCtx(metadata = UpdateMeta(tr, constraints)))
Update(pass, tr::Trace, constraints) where T = disablehooks(TraceCtx(pass = pass, metadata = UpdateMeta(tr, constraints)))

mutable struct RegenerateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    selection::UnconstrainedHierarchicalSelection
    RegenerateMeta(tr::T, sel::Vector{Address}) where T <: Trace = new{T}(tr, 
                                                                Address[], 
                                                                Union{Symbol, Pair}[],
                                                                selection(sel))
end
Regenerate(tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(metadata = RegenerateMeta(tr, sel)))
Regenerate(pass, tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(pass = pass, metadata = RegenerateMeta(tr, sel)))

mutable struct ScoreMeta{T <: Trace} <: Meta
    tr::T
    score::Float64
    visited::Vector{Address}
    Score(tr::T) where T <: Trace = new{T}(tr, 0.0, Address[])
end
Score(tr::Trace) = disablehooks(TraceCtx(metadata = Score(tr)))
Score(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = Score(tr)))

function reset_keep_constraints!(ctx::TraceCtx{M}) where M <: Meta
    ctx.metadata.tr = Trace()
    ctx.metadata.visited = Address[]
end

# --------------- OVERDUB -------------------- #

# ChoiceSites.
@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: UnconstrainedGenerateMeta, 
                                                  T <: Address}

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
    push!(ctx.metadata.visited, addr)
    return sample
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: GenerateMeta, 
                                                  T <: Address}

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)

    # Constrained..
    if haskey(ctx.metadata.constraints, addr)
        sample = ctx.metadata.constraints[addr]
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        ctx.metadata.tr.score += score
        push!(ctx.metadata.visited, addr)
        return sample

        # Unconstrained.
    else
        sample = rand(d)
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        push!(ctx.metadata.visited, addr)
        return sample
    end
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: ProposalMeta, 
                                                  T <: Address}

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)
    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
    ctx.metadata.tr.score += score
    push!(ctx.metadata.visited, addr)
    return sample

end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: RegenerateMeta, 
                                                  T <: Address}

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
    ctx.metadata.tr.chm[addr] = ChoiceSite(ret, score)

    # Visited
    push!(ctx.metadata.visited, addr)
    ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: UpdateMeta, 
                                                  T <: Address}

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
    ctx.metadata.tr.chm[addr] = ChoiceSite(ret, score)

    # Visited.
    push!(ctx.metadata.visited, addr)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args...) where {M <: ScoreMeta, 
                                                  T <: Address}
    # Get val.
    val = ctx.metadata.tr.chm[addr].value
    d = dist(args...)
    ctx.metadata.tr.score += logpdf(d, val)

    # Visited.
    push!(ctx.metadata.visited, addr)
    return val
end

# Function calls.

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UnconstrainedGenerateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = UnconstrainedGenerateMeta(Trace()))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                     call, 
                                     args..., 
                                     ret)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UpdateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = Update(Trace(), ctx.metadata.constraints[addr]))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args..., 
                                         ret)
    return ret
end

# Fallback - encounter some Cassette issues if we don't force this override for some Core functions.
@inline Cassette.fallback(ctx::TraceCtx, c::Function, args...) = c(args...)
