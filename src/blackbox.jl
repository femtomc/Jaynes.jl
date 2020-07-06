# This interface allows external calls/libraries to interact with the probabilistic tracing system. It defines new contextual primitives for overdubs - it requires that the user provide a specified logpdf.

macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex

        @inline function (tr::Jaynes.HierarchicalTrace)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            s = $argname(args...)
            tr.chm[addr] = Jaynes.ChoiceSite(logpdf($argname, args..., s), s)
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            # Constrained..
            if haskey(ctx.select.query, addr)
                s = ctx.select.query[addr]
                score = logpdf($argname, args..., s)
                ctx.tr.chm[addr] = Jaynes.ChoiceSite(score, s)
                ctx.tr.score += score
                return s

            # Unconstrained.
            else
                s = $argname(args...)
                score = logpdf($argname, args..., s)
                ctx.tr.chm[addr] = Jaynes.ChoiceSite(score, s)
                return s
            end
        end

        @inline function (ctx::Jaynes.ProposeContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            s = $argname(args...)
            score = logpdf($argname, args..., s)
            ctx.tr.chm[addr] = Jaynes.ChoiceSite(score, s)
            ctx.tr.score += score
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            # Check if in previous trace's choice map.
            in_prev_chm = haskey(ctx.prev.chm, addr)
            in_prev_chm && begin
                prev = ctx.prev.chm[addr]
                prev_val = prev.val
                prev_score = prev.score
            end

            # Check if in selection in meta.
            in_sel = haskey(ctx.select.query, addr)

            ret = $argname(args...)
            in_prev_chm && !in_sel && begin
                ret = prev_val
            end

            score = logpdf($argname, args..., ret)
            in_prev_chm && !in_sel && begin
                ctx.tr.score += score - prev_score
            end
            ctx.tr.chm[addr] = Jaynes.ChoiceSite(score, ret)

            # Visited.
            push!(ctx.visited, addr)

            ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            # Check if in previous trace's choice map.
            in_prev_chm = haskey(ctx.prev.chm, addr)
            in_prev_chm && begin
                prev = ctx.prev.chm[addr]
                prev_ret = prev.val
                prev_score = prev.score
            end

            # Check if in selection.
            in_selection = haskey(ctx.select.query, addr)

            # Ret.
            if in_selection
                ret = ctx.select.query[addr]
                push!(ctx.select_visited, addr)
            elseif in_prev_chm
                ret = prev_ret
            else
                ret = $argname(args...)
            end

            # Update.
            score = logpdf($argname, args..., ret)
            if in_prev_chm
                ctx.tr.score += score - prev_score
            elseif in_selection
                ctx.tr.score += score
            end
            ctx.tr.chm[addr] = Jaynes.ChoiceSite(score, ret)

            return ret
        end

        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            haskey(ctx.select.query, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.") && begin
                val = ctx.select.query[addr]
            end
            ctx.score += logpdf(d, val)
            return val

        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end

