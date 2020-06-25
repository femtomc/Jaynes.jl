Cassette.@context TraceCtx

# ------------------- META ----------------- #

abstract type Meta end

abstract type GenerateMeta <: Meta end
mutable struct UnconstrainedGenerateMeta{T <: Trace} <: GenerateMeta
    tr::T
    UnconstrainedGenerateMeta(tr::T) where T <: Trace = new{T}(tr)
end
Generate(tr::Trace) = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(tr)))
Generate(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGenerateMeta(tr)))

mutable struct ConstrainedGenerateMeta{T <: Trace, K <: ConstrainedSelection} <: GenerateMeta
    tr::T
    select::K
    ConstrainedGenerateMeta(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select)
end
Generate(tr::Trace, select::ConstrainedSelection) = disablehooks(TraceCtx(metadata = ConstrainedGenerateMeta(tr, select)))
Generate(pass, tr::Trace, select) = disablehooks(TraceCtx(pass = pass, metadata = ConstrainedGenerateMeta(tr, select)))

mutable struct ProposalMeta{T <: Trace} <: Meta
    tr::T
    ProposalMeta(tr::T) where T <: Trace = new{T}(tr)
end
Propose(tr::Trace) = disablehooks(TraceCtx(metadata = ProposalMeta(tr)))
Propose(pass, tr::Trace) = disablehooks(TraceCtx(pass = pass, metadata = ProposalMeta(tr)))

mutable struct UpdateMeta{T <: Trace, K <: ConstrainedSelection} <: Meta
    tr::T
    select_visited::Vector{Address}
    select::K
    UpdateMeta(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, Address[], select)
end

Update(tr::Trace, select) where T = disablehooks(TraceCtx(metadata = UpdateMeta(tr, select)))
Update(pass, tr::Trace, select) where T = disablehooks(TraceCtx(pass = pass, metadata = UpdateMeta(tr, select)))

abstract type RegenerateMeta <: Meta end
mutable struct UnconstrainedRegenerateMeta{T <: Trace, L <: UnconstrainedSelection} <: RegenerateMeta
    tr::T
    select::L
    visited::VisitedSelection
    function UnconstrainedRegenerateMeta(tr::T, sel::Vector{Address}) where T <: Trace
        un_sel = selection(sel)
        new{T, typeof(un_sel)}(tr, un_sel, VisitedSelection())
    end
    function UnconstrainedRegenerateMeta(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
        new{T, L}(tr, sel, VisitedSelection())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(metadata = UnconstrainedRegenerateMeta(tr, sel)))
Regenerate(tr::Trace, sel::UnconstrainedSelection) = disablehooks(TraceCtx(metadata = UnconstrainedRegenerateMeta(tr, sel)))
Regenerate(pass, tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedRegenerateMeta(tr, sel)))

mutable struct ConstrainedRegenerateMeta{T <: Trace, L <: UnconstrainedSelection, K <: ConstrainedSelection} <: Meta
    tr::T
    select::L
    observations::K
    visited::VisitedSelection
    function ConstrainedRegenerateMeta(tr::T, sel::Vector{Address}, obs::Vector{Tuple{K, P}}) where {T <: Trace, P, K <: Union{Symbol, Pair}} 
        un_sel = selection(sel)
        c_sel = selection(obs)
        new{T, typeof(un_sel), typeof(c_sel)}(tr, unsel, c_sel, VisitedSelection())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}, obs::Vector) = disablehooks(TraceCtx(metadata = ConstrainedRegenerateMeta(tr, sel, obs)))
Regenerate(pass, tr::Trace, sel::Vector{Address}, obs::Vector) = disablehooks(TraceCtx(pass = pass, metadata = ConstrainedRegenerateMeta(tr, sel, obs)))

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
    if haskey(ctx.metadata.select.query, addr)
        sample = ctx.metadata.select.query[addr]
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
                                  d::Distribution{K}) where {M <: UnconstrainedRegenerateMeta, 
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
    in_sel = haskey(ctx.metadata.select.query, addr)

    ret = rand(d)
    in_prev_chm && !in_sel && begin
        ret = prev_val
    end

    score = logpdf(d, ret)
    in_prev_chm && !in_sel && begin
        ctx.metadata.tr.score += score - prev_score
    end
    ctx.metadata.tr.chm[addr] = ChoiceSite(ret, score)

    # Visited.
    push!(ctx.metadata.visited, addr)

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
    in_selection = haskey(ctx.metadata.select.query, addr)

    # Ret.
    if in_selection
        ret = ctx.metadata.select.query[addr]
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
                                         args, 
                                         ret)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: ConstrainedGenerateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = ConstrainedGenerateMeta(Trace(), ctx.metadata.select[addr]))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args, 
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
                                         args, 
                                         ret)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UpdateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = Update(Trace(), ctx.metadata.select[addr]))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args, 
                                         ret)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UnconstrainedRegenerateMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = Regenerate(ctx.metadata.tr.chm[addr].trace, ctx.metadata.select[addr]))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args, 
                                         ret)
    ctx.metadata.tr.visited.tree[addr] = rec_ctx.metadata.tr.visited
    return ret
end

# Fallback - encounter some Cassette issues if we don't force this override for some Core functions.
@inline Cassette.fallback(ctx::TraceCtx, c::Function, args...) = c(args...)
