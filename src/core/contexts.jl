Cassette.@context TraceCtx

# ------------------- META ----------------- #

abstract type Meta end

abstract type GenerateMeta <: Meta end
mutable struct UnconstrainedGenerateMeta{T <: Trace} <: GenerateMeta
    tr::T
    UnconstrainedGenerateMeta(tr::T) where T <: Trace = new{T}(tr)
end
Generate(tr::Trace) = disablehooks(TraceCtx(pass = ignore_pass, metadata = UnconstrainedGenerateMeta(tr)))

mutable struct ConstrainedGenerateMeta{T <: Trace, K <: ConstrainedSelection} <: GenerateMeta
    tr::T
    select::K
    visited::VisitedSelection
    ConstrainedGenerateMeta(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, select, VisitedSelection())
end
Generate(tr::Trace, select::ConstrainedSelection) = disablehooks(TraceCtx(pass = ignore_pass, metadata = ConstrainedGenerateMeta(tr, select)))

mutable struct ProposalMeta{T <: Trace} <: Meta
    tr::T
    ProposalMeta(tr::T) where T <: Trace = new{T}(tr)
end
Propose(tr::Trace) = disablehooks(TraceCtx(pass = ignore_pass, metadata = ProposalMeta(tr)))

mutable struct UpdateMeta{T <: Trace, K <: ConstrainedSelection} <: Meta
    prev::T
    tr::T
    select::K
    visited::VisitedSelection
    UpdateMeta(tr::T, select::K) where {T <: Trace, K <: ConstrainedSelection} = new{T, K}(tr, Trace(), select, VisitedSelection())
end

Update(tr::Trace, select) where T = disablehooks(TraceCtx(pass = ignore_pass, metadata = UpdateMeta(tr, select)))

abstract type RegenerateMeta <: Meta end
mutable struct UnconstrainedRegenerateMeta{T <: Trace, L <: UnconstrainedSelection} <: RegenerateMeta
    prev::T
    tr::T
    select::L
    visited::VisitedSelection
    function UnconstrainedRegenerateMeta(tr::T, sel::Vector{Address}) where T <: Trace
        un_sel = selection(sel)
        new{T, typeof(un_sel)}(tr, Trace(), un_sel, VisitedSelection())
    end
    function UnconstrainedRegenerateMeta(tr::T, sel::L) where {T <: Trace, L <: UnconstrainedSelection}
        new{T, L}(tr, Trace(), sel, VisitedSelection())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}) = disablehooks(TraceCtx(pass = ignore_pass, metadata = UnconstrainedRegenerateMeta(tr, sel)))
Regenerate(tr::Trace, sel::UnconstrainedSelection) = disablehooks(TraceCtx(pass = ignore_pass, metadata = UnconstrainedRegenerateMeta(tr, sel)))

mutable struct ConstrainedRegenerateMeta{T <: Trace, L <: UnconstrainedSelection, K <: ConstrainedSelection} <: Meta
    prev::T
    tr::T
    select::L
    observations::K
    visited::VisitedSelection
    function ConstrainedRegenerateMeta(tr::T, sel::Vector{Address}, obs::Vector{Tuple{K, P}}) where {T <: Trace, P, K <: Union{Symbol, Pair}} 
        un_sel = selection(sel)
        c_sel = selection(obs)
        new{T, typeof(un_sel), typeof(c_sel)}(tr, Trace(), unsel, c_sel, VisitedSelection())
    end
end
Regenerate(tr::Trace, sel::Vector{Address}, obs::Vector) = disablehooks(TraceCtx(pass = ignore_pass, metadata = ConstrainedRegenerateMeta(tr, sel, obs)))

mutable struct ScoreMeta{K <: ConstrainedSelection} <: Meta
    score::Float64
    select::K
    function Score(obs::Vector{Tuple{K, P}}) where {P, K <: Union{Symbol, Pair}}
        c_sel = selection(obs)
        new{typeof(c_sel)}(0.0, c_sel)
    end
    Score(obs::K) where {K <: ConstrainedSelection} = new{K}(0.0, obs)
end
Score(obs::Vector) = disablehooks(TraceCtx(pass = ignore_pass, metadata = Score(obs)))
Score(obs::ConstrainedSelection) = disablehooks(TraceCtx(pass = ignore_pass, metadata = Score(sel)))

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

    if haskey(ctx.metadata.select.query, addr)
        sample = ctx.metadata.select.query[addr]
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        ctx.metadata.tr.score += score
        push!(ctx.metadata.visited, addr)
    else
        sample = rand(d)
        score = logpdf(d, sample)
        ctx.metadata.tr.chm[addr] = ChoiceSite(sample, score)
        push!(ctx.metadata.visited, addr)
    end
    return sample
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
    in_prev_chm = haskey(ctx.metadata.prev.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.prev.chm[addr]
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
    in_prev_chm = haskey(ctx.metadata.prev.chm, addr)
    in_prev_chm && begin
        prev = ctx.metadata.prev.chm[addr]
        prev_ret = prev.val
        prev_score = prev.score
    end

    # Check if in selection.
    in_selection = haskey(ctx.metadata.select.query, addr)

    # Ret.
    if in_selection
        ret = ctx.metadata.select.query[addr]
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

    push!(ctx.metadata.visited, addr)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  d::Distribution{K}) where {M <: ScoreMeta, 
                                                             T <: Address,
                                                             K}

    haskey(ctx.metadata.select.query, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.") && begin
        val = ctx.metadata.select.query[addr]
    end
    ctx.metadata.score += logpdf(d, val)
    return val
end

# Call sites.

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UnconstrainedGenerateMeta, 
                                                  T <: Address}
    ug_ctx = similarcontext(ctx; metadata = UnconstrainedGenerateMeta(Trace()))
    ret = recurse(ug_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(ug_ctx.metadata.tr, 
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

    cg_ctx = similarcontext(ctx; metadata = ConstrainedGenerateMeta(Trace(), ctx.metadata.select[addr]))
    ret = recurse(cg_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(cg_ctx.metadata.tr, 
                                         call, 
                                         args, 
                                         ret)
    ctx.metadata.visited.tree[addr] = cg_ctx.metadata.visited
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: ProposalMeta, 
                                                  T <: Address}

    p_ctx = similarcontext(ctx; metadata = Propose(Trace()))
    ret = recurse(p_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(p_ctx.metadata.tr, 
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

    u_ctx = similarcontext(ctx; metadata = Update(ctx.metadata.prev.chm[addr].trace, ctx.metadata.select[addr]))
    ret = recurse(u_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(u_ctx.metadata.tr, 
                                         call, 
                                         args, 
                                         ret)
    ctx.metadata.visited.tree[addr] = u_ctx.metadata.visited
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UnconstrainedRegenerateMeta, 
                                                  T <: Address}

    ur_ctx = similarcontext(ctx; metadata = Regenerate(ctx.metadata.prev.chm[addr].trace, ctx.metadata.select[addr]))
    ret = recurse(ur_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(ur_ctx.metadata.tr, 
                                         call, 
                                         args, 
                                         ret)
    ctx.metadata.visited.tree[addr] = ur_ctx.metadata.visited
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: ScoreMeta, 
                                                  T <: Address}

    s_ctx = similarcontext(ctx; metadata = Score(ctx.metadata.select[addr]))
    ret = recurse(s_ctx, call, args...)
    return ret
end

# Fallback - encounter some Cassette issues if we don't force this override for some Core functions.
@inline Cassette.fallback(ctx::TraceCtx, c::Function, args...) = c(args...)
