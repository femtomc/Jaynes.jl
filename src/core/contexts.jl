Cassette.@context TraceCtx

# ------------------- META ----------------- #

abstract type Meta end

struct UnconstrainedGenerateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    UnconstrainedGenerateMeta(tr::T) where T <: Trace = new{T}(tr, Address[])
end
Generate(tr::Trace) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGenerateMeta(tr)))

struct ConstrainedGenerateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    select::ConstrainedHierarchicalSelection
    ConstrainedGenerateMeta(tr::T, select::ConstrainedHierarchicalSelection) where T <: Trace = new{None}(tr, Address[], Union{Symbol,Pair}[], select)
end
Generate(tr::Trace, select::ConstrainedHierarchicalSelection) = disablehooks(TraceCtx(metadata = ConstrainedGenerateMeta(tr, select)))
Generate(pass, tr::Trace, select) = disablehooks(TraceCtx(pass = pass, metadata = ConstrainedGenerateMeta(tr, select)))

struct ProposalMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    ProposalMeta(tr::T) where T <: Trace = new{T}(tr, Address[])
end
Propose(tr::Trace) = disablehooks(TraceCtx(metadata = ProposalMeta(tr)))
Propose(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = ProposalMeta(tr)))

struct UpdateMeta{T <: Trace} <: Meta
    tr::T
    visited::Vector{Address}
    select_visited::Vector{Address}
    select::ConstrainedHierarchicalSelection
    UpdateMeta(tr::T, select::ConstrainedHierarchicalSelection) where T <: Trace = new{T}(tr, Address[], Union{Symbol, Pair}[], select)
end
Update(tr::Trace, select) where T = disablehooks(TraceCtx(metadata = UpdateMeta(tr, select)))
Update(pass, tr::Trace, select) where T = disablehooks(TraceCtx(pass = pass, metadata = UpdateMeta(tr, select)))

struct RegenerateMeta{T <: Trace} <: Meta
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

function reset_keep_select!(ctx::TraceCtx{M}) where M <: Meta
    ctx.metadata.tr = Trace()
    ctx.metadata.visited = Address[]
end

# ------------------ OVERDUB -------------------- #

# Choice sites.

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
                                  args...) where {M <: ConstrainedGenerateMeta, 
                                                  T <: Address}

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    d = dist(args...)

    # Constrained..
    if haskey(ctx.metadata.select, addr)
        sample = ctx.metadata.select[addr]
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
    in_sel = haskey(ctx.metadata.select, addr)

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

    # Check if in selection.
    in_selection = haskey(ctx.metadata.select, addr)

    # Ret.
    d = dist(args...)
    if in_selection
        ret = ctx.metadata.select[addr]
        push!(ctx.metadata.select_visited, addr)
    elseif in_prev_chm
        ret = prev_ret
    else
        ret = rand(d)
    end

    # Update.
    score = logpdf(d, ret)
    if in_prev_chm
        ctx.metadata.tr.score += score - prev_score
    elseif in_selection
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

# Call sites.

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
                                  args...) where {M <: ConstrainedGenerateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = ConstrainedGenerateMeta(Trace(), ctx.metadata.select.tree[addr]))
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
                                  args...) where {M <: ProposalMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = Propose(Trace()))
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

    rec_ctx = similarcontext(ctx; metadata = Update(Trace(), ctx.metadata.select.tree[addr]))
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
                                  args...) where {M <: RegenerateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = Regenerate(Trace(), ctx.metadata.select.tree[addr]))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args..., 
                                         ret)
    return ret
end

# Fallback - encounter some Cassette issues if we don't force this override for some Core functions.
@inline Cassette.fallback(ctx::TraceCtx, c::Function, args...) = c(args...)
