Cassette.@context TraceCtx

# ------------------- META ----------------- #

abstract type Meta end

mutable struct UnconstrainedGenerateMeta{T <: Trace} <: Meta
    tr::T
    UnconstrainedGenerateMeta(tr::T) where T <: Trace = new{T}(tr)
end
Generate(tr::Trace) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGenerateMeta(tr)))

mutable struct ConstrainedGenerateMeta{T <: Trace} <: Meta
    tr::T
    select::ConstrainedHierarchicalSelection
    ConstrainedGenerateMeta(tr::T, select::ConstrainedHierarchicalSelection) where T <: Trace = new{T}(tr, select)
end
Generate(tr::Trace, select::ConstrainedHierarchicalSelection) = disablehooks(TraceCtx(metadata = ConstrainedGenerateMeta(tr, select)))
Generate(pass, tr::Trace, select) = disablehooks(TraceCtx(pass = pass, metadata = ConstrainedGenerateMeta(tr, select)))

mutable struct ProposalMeta{T <: Trace} <: Meta
    tr::T
    ProposalMeta(tr::T) where T <: Trace = new{T}(tr)
end
Propose(tr::Trace) = disablehooks(TraceCtx(metadata = ProposalMeta(tr)))
Propose(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = ProposalMeta(tr)))

mutable struct UpdateMeta{T <: Trace} <: Meta
    tr::T
    select_visited::Vector{Address}
    select::ConstrainedHierarchicalSelection
    UpdateMeta(tr::T, select::ConstrainedHierarchicalSelection) where T <: Trace = new{T}(tr, Address[], select)
end

Update(tr::Trace, select) where T = disablehooks(TraceCtx(metadata = UpdateMeta(tr, select)))
Update(pass, tr::Trace, select) where T = disablehooks(TraceCtx(pass = pass, metadata = UpdateMeta(tr, select)))

mutable struct RegenerateMeta{T <: Trace} <: Meta
    tr::T
    selection::UnconstrainedHierarchicalSelection
    RegenerateMeta(tr::T, sel::Vector{Address}) where T <: Trace = new{T}(tr, 
                                                                          selection(sel))
end
Regenerate(tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(metadata = RegenerateMeta(tr, sel)))
Regenerate(pass, tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(pass = pass, metadata = RegenerateMeta(tr, sel)))

mutable struct ScoreMeta{T <: Trace} <: Meta
    tr::T
    score::Float64
    Score(tr::T) where T <: Trace = new{T}(tr, 0.0)
end
Score(tr::Trace) = disablehooks(TraceCtx(metadata = Score(tr)))
Score(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = Score(tr)))

reset!(ctx::TraceCtx{M}) where M <: Meta = ctx.metadata.tr = Trace()

# ------------------ OVERDUB -------------------- #

# Choice sites.

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: UnconstrainedGenerateMeta, 
                                                             T <: Address,
                                                             K}

    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
    return sample
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: ConstrainedGenerateMeta, 
                                                             T <: Address,
                                                             K}

    # Constrained..
    if haskey(ctx.metadata.select, addr)
        sample = ctx.metadata.select[addr]
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        ctx.metadata.tr.score += score
        return sample

        # Unconstrained.
    else
        sample = rand(d)
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        return sample
    end
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: ProposalMeta, 
                                                             T <: Address, 
                                                             K}

    sample = rand(d)
    score = logpdf(d, sample)
    ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
    ctx.metadata.tr.score += score
    return sample

end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: RegenerateMeta, 
                                                             T <: Address,
                                                             K}

    # Check if in previous trace's choice map.
    in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.tr.chm[addr]
        prev_val = prev.val
        prev_score = prev.score
    end

    # Check if in selection in meta.
    in_sel = haskey(ctx.metadata.select, addr)

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
    ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: UpdateMeta, 
                                                             T <: Address,
                                                             K}

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

    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: ScoreMeta, 
                                                             T <: Address,
                                                             K}
    # Get val.
    val = ctx.metadata.tr.chm[addr].value
    ctx.metadata.tr.score += logpdf(d, val)

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