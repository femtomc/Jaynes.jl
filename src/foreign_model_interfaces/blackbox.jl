# This interface allows external calls/libraries to interact with the probabilistic tracing system. It defines new contextual primitives for overdubs - it requires that the user provide a specified logpdf.

macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex

        @inline function (tr::Jaynes.HierarchicalTrace)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            s = $argname(args...)
            Jaynes.add_choice!(tr, addr, Jaynes.ChoiceSite(logpdf($argname, args..., s), s))
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx.visited, addr)
            if Jaynes.has_query(ctx.select, addr)
                s = Jaynes.get_query(ctx.select, addr)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(tr, addr, Jaynes.ChoiceSite(score, s))
                increment!(ctx, score)
            else
                s = $argname(args...)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            end
            return s
        end

        @inline function (ctx::Jaynes.ProposeContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            s = $argname(args...)
            score = logpdf($argname, args..., s)
            Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            increment!(ctx, score)
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            visit!(ctx.visited, addr)
            in_prev_chm = has_choice(ctx.prev, addr)
            in_sel = has_query(ctx.select, addr)
            if in_prev_chm
                prev = get_choice(ctx.prev, addr)
                if in_sel
                    ret = $argname(args...)
                    add_choice!(ctx.discard, addr, prev)
                else
                    ret = prev.val
                end
            end
            score = logpdf($argname, args..., ret)
            if in_prev_chm && !in_sel
                increment!(ctx, score - prev.score)
            end
            add_choice!(ctx.tr, addr, ChoiceSite(score, ret))
            return ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     $argname::$name,
                                                     args...) where {T <: Jaynes.Address, K}
            in_prev_chm = Jaynes.has_choice(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.get_choice(ctx.prev, addr)
                prev_ret = prev.val
                prev_score = prev.score
            end
            in_selection = Jaynes.has_query(ctx.select, addr)
            if in_selection
                ret = Jaynes.get_query(ctx.select, addr)
                in_prev_chm && begin
                    Jaynes.add_choice!(ctx.discard, addr, prev)
                end
                Jaynes.visit!(ctx.visited, addr)
            elseif in_prev_chm
                ret = prev_ret
            else
                ret = $argname(args...)
            end
            score = logpdf($argname, args..., ret)
            if in_prev_chm
                increment!(ctx, score - prev_score)
            elseif in_selection
                increment!(ctx, score)
            end
            Jaynes.add_choice!(ctx.tr, addr, ChoiceSite(score, ret))
            return ret
        end


        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.has_query(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
            val = Jaynes.get_query(ctx.select, addr)
            increment!(ctx, logpdf(d, val))
            return val

        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
