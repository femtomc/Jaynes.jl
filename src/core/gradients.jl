struct SiteGradients{K}
    sample::Float64
    gs::K
end

struct CallGradients
    map::Dict{Address, CallGradients}
    gs::Dict{Address, Vector{SiteGradients}}
    gs_args::IdDict
end

# Tracks learnable parameters.
struct CallLearnables
    map::Dict{Address, CallLearnables}
    vals::Dict{Address, Union{Number, AbstractArray}}
end

# Construct the dependency graph at runtime.
abstract type RuntimeAnalysis <: Analysis end
mutable struct GradientAnalysis <: RuntimeAnalysis
    map::Dict{Address, GradientAnalysis}
    parents::Dict{Address, Vector{Address}}
    tracker::IdDict{Union{Number, AbstractArray}, Address}
end

mutable struct UnconstrainedGradientMeta <: Meta
    visited::Vector{Address}
    ga::GradientAnalysis
    learned::CallLearnables
    cgs::CallGradients
    tr::Trace
    loss::Float64
    args::Tuple
    fn::Function
    ret::Any
    UnconstrainedGradientMeta() = new(Address[], 
                                      LearnableUnconstrainedSelection(),
                                      Dict{Address, Number}(), 
                                      Dict{Address, Tuple}(), 
                                      IdDict{Float64, Address}(), 
                                      Dict{Address, Vector{Address}}(),
                                      Trace(),
                                      0.0)
    UnconstrainedGradientMeta(tr::Trace) = new(Address[], 
                                               LearnableUnconstrainedSelection(),
                                               Dict{Address, Number}(), 
                                               Dict{Address, Tuple}(), 
                                               IdDict{Float64, Address}(), 
                                               Dict{Address, Vector{Address}}(),
                                               tr,
                                               0.0)
end
Gradient() = disablehooks(TraceCtx(metadata = UnconstrainedGradientMeta()))
Gradient(pass::Cassette.AbstractPass) = disablehooks(TraceCtx(pass = pass, metadata = UnconstrainedGradientMeta()))

# ------------------ OVERDUB -------------------- #

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: UnconstrainedGradientMeta, 
                                               T <: Address}

    # Build dependency graph.
    passed_in = filter(args) do a 
        if a in keys(ctx.metadata.sel.tracker)
            k = ctx.metadata.sel.tracker[a]
            if haskey(ctx.metadata.sel.parents, addr) && !(k in ctx.metadata.sel.parents[addr])
                push!(ctx.metadata.sel.parents[addr], k)
            else
                ctx.metadata.sel.parents[addr] = [k]
            end
            true
        else
            false
        end
    end

    # If args are in set of learned parameters, replace with learned versions.
    args = map(args) do a
        if haskey(ctx.metadata.sel.tracker, a)
            k = ctx.metadata.sel.tracker[a]
            if haskey(ctx.metadata.sel.learned, k)
                return ctx.metadata.sel.learned[k][1]
            else
                a
            end
        else
            a
        end
    end

    passed_in = IdDict{Any, Address}(map(passed_in) do a
                                         k = ctx.metadata.sel.tracker[a]
                                         a => k
                                     end)

    # Check trace for choice map.
    !haskey(ctx.metadata.tr.chm, addr) && error("UnconstrainedGradientMeta: toplevel function call has address space which does not match the training trace.")
    sample = ctx.metadata.tr.chm[addr].val
    ctx.metadata.sel.tracker[sample] = addr

    # Gradients
    gs = Flux.gradient((s, a) -> (loss = -logpdf(dist(a...), s);
                                  ctx.metadata.loss += loss;
                                  loss), sample, args)

    args_arr = Pair{Address, Float64}[]
    map(enumerate(args)) do (i, a)
        haskey(passed_in, a) && begin
            push!(args_arr, passed_in[a] => gs[2][i])
        end
    end

    if !isempty(args_arr)
        grads = SiteGradients(gs[1], Dict(args_arr...))

        # Push grads to parents.
        map(ctx.metadata.sel.parents[addr]) do p
            p in keys(ctx.metadata.sel.learned) && begin
                if haskey(ctx.metadata.sel.gradients, p)
                    push!(ctx.metadata.sel.gradients[p], grads)

                else
                    ctx.metadata.gradients[p] = [grads]
                end
            end
        end
    end

    push!(ctx.metadata.visited, addr)
    return sample
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(learnable), 
                                  addr::T,
                                  lit::K) where {M <: UnconstrainedGradientMeta, 
                                                 T <: Address,
                                                 K <: Union{Number, AbstractArray}}

    # Check for support errors.
    addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

    # Check if value in trainable.
    if haskey(ctx.metadata.sel.learned, addr)
        ret = ctx.metadata.sel.learned[addr][1]
    else
        ret = lit
    end

    # Track.
    ctx.metadata.tracker[ret] = addr
    !haskey(ctx.metadata.sel.learned, addr) && begin
        ctx.metadata.sel.learned[addr] = [ret]
    end

    push!(ctx.metadata.visited, addr)
    return ret
end

@inline function Cassette.overdub(ctx::TraceCtx{M},
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where {M <: UnconstrainedGradientMeta, 
                                                  T <: Address}

    rec_ctx = similarcontext(ctx; metadata = UnconstrainedGradientMeta())
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args..., 
                                         ret)
    ctx.metadata.sel.map[addr] = rec_ctx.metadata.sel
    return ret
end

# ------------------ END OVERDUB ------------------- #

import Flux: update!

function update!(opt, ctx::TraceCtx{M}) where M <: UnconstrainedGradientMeta
    for (k, val) in ctx.metadata.sel.learned
        if haskey(ctx.metadata.gradients, k)
            averaged = Dict{Address, Float64}()
            map(ctx.metadata.gradients[k]) do g
                for k in keys(g.gs)
                    !haskey(averaged, k) && begin
                        averaged[k] = 0.0
                    end
                    averaged[k] += g.gs[k]
                end
            end
            ctx.metadata.sel.learned[k] = update!(opt, val, [averaged[k]])
        end
    end
    ctx.metadata.gradients = Dict{Address, Union{Number, AbstractArray}}()
    ctx.metadata.visited = Address[]
    ctx.metadata.tracker = IdDict{Union{Number, AbstractArray}, Address}()
end

function train!(opt, fn::Function, args::Tuple, trs::Vector{Trace})
    ctx = Gradient()
    losses = Vector{Float64}(undef, length(trs))
    map(enumerate(trs)) do (i, tr)
        ctx.metadata.tr = tr
        ctx, _ = trace(ctx, fn, args)
        Jaynes.update!(opt, ctx)
        losses[i] = ctx.metadata.loss
        ctx.metadata.loss = 0.0
    end
    ctx.metadata.fn = fn
    ctx.metadata.args = args
    return ctx, losses
end
