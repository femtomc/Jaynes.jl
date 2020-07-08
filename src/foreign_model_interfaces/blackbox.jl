# This interface allows external calls/libraries to interact with the probabilistic tracing system. It defines new contextual primitives for overdubs - it requires that the user provide a specified logpdf.

macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex

        @inline function (tr::Jaynes.HierarchicalTrace)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            s = $argname(args...)
            Jaynes.set_choice!(tr, addr, Jaynes.ChoiceSite(logpdf($argname, args..., s), s))
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            # Constrained..
            if Jaynes.has_query(ctx.select, addr)
                s = Jaynes.get_query(ctx.select, addr)
                score = logpdf($argname, args..., s)
                Jaynes.set_choice!(tr, addr, Jaynes.ChoiceSite(score, s))
                ctx.tr.score += score

            # Unconstrained.
            else
                s = $argname(args...)
                score = logpdf($argname, args..., s)
                Jaynes.set_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            end
            Jaynes.visit!(ctx.visited, addr)
            return s
        end

        @inline function (ctx::Jaynes.ProposeContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            s = $argname(args...)
            score = logpdf($argname, args..., s)
            Jaynes.set_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            ctx.tr.score += score
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            # Check if in previous trace's choice map.
            in_prev_chm = Jaynes.has_choice(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.get_choice(ctx.prev, addr)
                prev_val = prev.val
                prev_score = prev.score
            end

            # Check if in selection in meta.
            in_sel = Jaynes.has_query(ctx.select, addr)

            ret = $argname(args...)
            in_prev_chm && !in_sel && begin
                ret = prev_val
            end

            score = logpdf($argname, args..., ret)
            in_prev_chm && !in_sel && begin
                ctx.tr.score += score - prev_score
            end
            Jaynes.set_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, ret))

            # Visited.
            Jaynes.visit!(ctx.visited, addr)
            return ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     $argname::$name,
                                                     args...) where {T <: Jaynes.Address, K}
            # Check if in previous trace's choice map.
            in_prev_chm = Jaynes.has_choice(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.get_choice(ctx.prev, addr)
                prev_ret = prev.val
                prev_score = prev.score
            end

            # Check if in selection.
            in_selection = Jaynes.has_query(ctx.select, addr)

            # Ret.
            if in_selection
                ret = Jaynes.get_query(ctx.select, addr)
                in_prev_chm && begin
                    Jaynes.set_choice!(ctx.discard, addr, prev)
                end
                Jaynes.visit!(ctx.visited, addr)
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
            Jaynes.set_choice!(ctx.tr, addr, ChoiceSite(score, ret))

            return ret
        end


        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}

            Jaynes.has_query(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
            val = Jaynes.get_query(ctx.select, addr)
            ctx.score += logpdf(d, val)
            return val

        end
    end

    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end

