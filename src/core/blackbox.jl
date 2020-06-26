# This interface allows external calls/libraries to interact with the probabilistic tracing system. It defines new contextual primitives for overdubs - it requires that the user provide a specified logpdf.

macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex
        @inline function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.UnconstrainedGenerateMeta, 
                                                          T <: Jaynes.Address,
                                                          K}

            sample = $argname(args...)
            score = logpdf($argname, args..., sample)
            ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(sample, score)
            return sample
        end

        function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.ConstrainedGenerateMeta, 
                                                          T <: Jaynes.Address,
                                                          K}

            # Constrained..
            if haskey(ctx.metadata.select.query, addr)
                sample = ctx.metadata.select.query[addr]
                score = logpdf($argname, args..., sample)
                ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(sample, score)
                ctx.metadata.tr.score += score
                return sample

            # Unconstrained.
            else
                sample = $argname(args...)
                score = logpdf($argname, args..., sample)
                ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(sample, score)
                return sample
            end
        end

        @inline function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.ProposalMeta, 
                                                          T <: Jaynes.Address, 
                                                          K}

            sample = $argname(args...)
            score = logpdf($argname, args..., sample)
            ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(sample, score)
            ctx.metadata.tr.score += score
            return sample

        end

        @inline function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.UnconstrainedRegenerateMeta, 
                                                          T <: Jaynes.Address,
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

            ret = $argname(args...)
            in_prev_chm && !in_sel && begin
                ret = prev_val
            end

            score = logpdf($argname, args..., ret)
            in_prev_chm && !in_sel && begin
                ctx.metadata.tr.score += score - prev_score
            end
            ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(ret, score)

            # Visited.
            push!(ctx.metadata.visited, addr)

            ret
        end

        @inline function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.UpdateMeta, 
                                                          T <: Jaynes.Address,
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
                ret = $argname(args...)
            end

            # Update.
            score = logpdf($argname, args..., ret)
            if in_prev_chm
                ctx.metadata.tr.score += score - prev_score
            elseif in_selection
                ctx.metadata.tr.score += score
            end
            ctx.metadata.tr.chm[addr] = Jaynes.ChoiceSite(ret, score)

            return ret
        end

        @inline function Jaynes.overdub(ctx::Jaynes.TraceCtx{M}, 
                                          call::typeof(rand), 
                                          addr::T, 
                                          $argname::$name,
                                          args...) where {M <: Jaynes.ScoreMeta, 
                                                          T <: Jaynes.Address,
                                                          K}
            # Get val.
            val = ctx.metadata.tr.chm[addr].value
            ctx.metadata.tr.score += logpdf($argname, args..., val)

            return val
        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
