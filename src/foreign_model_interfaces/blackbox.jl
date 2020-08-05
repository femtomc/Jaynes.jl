macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex

        @inline function (ctx::Jaynes.SimulateContext)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            Jaynes.visit!(ctx.visited, addr)
            s = $argname(args...)
            Jaynes.add_choice!(ctx, addr, Jaynes.ChoiceSite(logpdf($argname, args..., s), s))
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx.visited, addr)
            if Jaynes.has_top(ctx.select, addr)
                s = Jaynes.get_top(ctx.select, addr)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(tr, addr, Jaynes.ChoiceSite(score, s))
                Jaynes.increment!(ctx, score)
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
            Jaynes.increment!(ctx, score)
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx.visited, addr)
            in_prev_chm = Jaynes.has_top(ctx.prev, addr)
            in_sel = Jaynes.has_top(ctx.select, addr)
            if in_prev_chm
                prev = Jaynes.get_top(ctx.prev, addr)
                if in_sel
                    ret = $argname(args...)
                    Jaynes.add_choice!(ctx.discard, addr, prev)
                else
                    ret = prev.val
                end
            end
            score = logpdf($argname, args..., ret)
            if in_prev_chm && !in_sel
                Jaynes.increment!(ctx, score - prev.score)
            end
            Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, ret))
            return ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     $argname::$name,
                                                     args...) where {T <: Jaynes.Address, K}
            in_prev_chm = Jaynes.has_top(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.get_top(ctx.prev, addr)
                prev_ret = prev.val
                prev_score = prev.score
            end
            in_selection = Jaynes.has_top(ctx.select, addr)
            if in_selection
                ret = Jaynes.get_top(ctx.select, addr)
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
                Jaynes.increment!(ctx, score - prev_score)
            elseif in_selection
                Jaynes.increment!(ctx, score)
            end
            Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, ret))
            return ret
        end


        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.has_top(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
            val = Jaynes.get_top(ctx.select, addr)
            Jaynes.increment!(ctx, logpdf(d, val))
            return val

        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end

macro primitive(inject, ex)
    @capture(inject, nm_ -> injbody_) || error("InjectionError: defining an injection requires a valid lambda expression.")
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    inject_name = gensym("inject")
    expr = quote
        $ex
        $inject_name = $nm -> $injbody
        @inline function (ctx::Jaynes.SimulateContext)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            Jaynes.visit!(ctx.visited, addr)
            s = $argname(args...)
            s = $inject_name(s)
            Jaynes.add_choice!(ctx, addr, Jaynes.ChoiceSite(logpdf($argname, args..., s), s))
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx.visited, addr)
            if Jaynes.has_top(ctx.select, addr)
                s = Jaynes.get_top(ctx.select, addr)
                s = $inject_name(s)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(tr, addr, Jaynes.ChoiceSite(score, s))
                Jaynes.increment!(ctx, score)
            else
                s = $argname(args...)
                s = $inject_name(s)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            end
            return s
        end

        @inline function (ctx::Jaynes.ProposeContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            s = $argname(args...)
            s = $inject_name(s)
            score = logpdf($argname, args..., s)
            Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, s))
            Jaynes.increment!(ctx, score)
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            visit!(ctx.visited, addr)
            in_prev_chm = has_top(ctx.prev, addr)
            in_sel = has_top(ctx.select, addr)
            if in_prev_chm
                prev = get_top(ctx.prev, addr)
                if in_sel
                    ret = $argname(args...)
                    ret = $inject_name(ret)
                    add_choice!(ctx.discard, addr, prev)
                else
                    ret = prev.val
                end
            end
            score = logpdf($argname, args..., ret)
            if in_prev_chm && !in_sel
                Jaynes.increment!(ctx, score - prev.score)
            end
            add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, ret))
            return ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     $argname::$name,
                                                     args...) where {T <: Jaynes.Address, K}
            in_prev_chm = Jaynes.has_top(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.get_top(ctx.prev, addr)
                prev_ret = prev.val
                prev_score = prev.score
            end
            in_selection = Jaynes.has_top(ctx.select, addr)
            if in_selection
                ret = Jaynes.get_top(ctx.select, addr)
                in_prev_chm && begin
                    Jaynes.add_choice!(ctx.discard, addr, prev)
                end
                Jaynes.visit!(ctx.visited, addr)
            elseif in_prev_chm
                ret = prev_ret
            else
                ret = $argname(args...)
                ret = $inject_name(ret)
            end
            score = logpdf($argname, args..., ret)
            if in_prev_chm
                Jaynes.increment!(ctx, score - prev_score)
            elseif in_selection
                Jaynes.increment!(ctx, score)
            end
            Jaynes.add_choice!(ctx.tr, addr, Jaynes.ChoiceSite(score, ret))
            return ret
        end


        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.has_top(ctx.select, addr) || error("ScoreError: constrained selection must provide constraints for all possible addresses in trace. Missing at address $addr.")
            val = Jaynes.get_top(ctx.select, addr)
            Jaynes.increment!(ctx, logpdf(d, val))
            return val

        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
