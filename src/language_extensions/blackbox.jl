macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex

        @inline function (ctx::Jaynes.SimulateContext)(call::typeof(rand), addr::Jaynes.Address, $argname::$name, args...)
            Jaynes.visit!(ctx.visited, addr)
            s = $argname(args...)
            Jaynes.add_choice!(ctx, addr, logpdf($argname, args..., s), s)
            return s
        end

        function (ctx::Jaynes.GenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx.visited, addr)
            if Jaynes.haskey(ctx.target, addr)
                s = Jaynes.getindex(ctx.target, addr)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(ctx, addr, score, s)
                Jaynes.increment!(ctx, score)
            else
                s = $argname(args...)
                score = logpdf($argname, args..., s)
                Jaynes.add_choice!(ctx, addr, score, s)
            end
            return s
        end

        @inline function (ctx::Jaynes.ProposeContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            s = $argname(args...)
            score = logpdf($argname, args..., s)
            Jaynes.add_choice!(ctx, addr, score, s)
            Jaynes.increment!(ctx, score)
            return s
        end

        @inline function (ctx::Jaynes.RegenerateContext)(call::typeof(rand), addr::T, $argname::$name, args...) where {T <: Jaynes.Address, K}
            Jaynes.visit!(ctx, addr)
            in_prev_chm = Jaynes.has_value(Jaynes.get_trace(ctx.prev), addr)
            in_sel = Jaynes.haskey(ctx.target, addr)

            if in_prev_chm
                prev = Jaynes.get_sub(Jaynes.get_trace(ctx.prev), addr)
            end

            if in_sel && in_prev_chm
                ret = rand(d)
                Jaynes.set_sub!(ctx.discard, addr, prev)
            elseif in_prev_chm
                ret = prev.val
            else
                ret = rand(d)
            end

            score = logpdf(d, ret)
            in_prev_chm && Jaynes.increment!(ctx, score - Jaynes.get_score(prev))
            Jaynes.add_choice!(ctx, addr, score, ret)
            return ret
        end

        @inline function (ctx::Jaynes.UpdateContext)(call::typeof(rand), 
                                                     addr::T, 
                                                     $argname::$name,
                                                     args...) where {T <: Jaynes.Address, K}
            in_prev_chm = Jaynes.haskey(ctx.prev, addr)
            in_prev_chm && begin
                prev = Jaynes.getindex(ctx.prev, addr)
                prev_ret = prev.val
                prev_score = prev.score
            end
            in_target = Jaynes.haskey(ctx.target, addr)
            if in_target
                ret = Jaynes.getindex(ctx.target, addr)
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
            elseif in_target
                Jaynes.increment!(ctx, score)
            end
            Jaynes.add_choice!(ctx, addr, score, ret)
            return ret
        end


        @inline function (ctx::Jaynes.ScoreContext)(call::typeof(rand), 
                                                    addr::T, 
                                                    $argname::$name, 
                                                    args...) where {T <: Jaynes.Address, K}
            Jaynes.haskey(ctx.target, addr) || error("ScoreError: constrained target must provide constraints for all possible addresses in trace. Missing at address $addr.")
            val = Jaynes.getindex(ctx.target, addr)
            Jaynes.increment!(ctx, logpdf($argname, args..., val))
            return val

        end

        @inline function (ctx::Jaynes.ParameterBackpropagateContext)(call::typeof(rand), 
                                                                     addr::T, 
                                                                     $argname::$name,
                                                                     args...) where {T <: Jaynes.Address, K}
            if Jaynes.haskey(ctx.fixed, addr)
                s = Jaynes.getindex(ctx.fixed, addr)
            else
                s = Jaynes.get_value(Jaynes.get_sub(ctx.call, addr))
            end
            Jaynes.increment!(ctx, logpdf($argname, args..., s))
            return s
        end

        @inline function (ctx::Jaynes.ChoiceBackpropagateContext)(call::typeof(rand), 
                                                                  addr::T, 
                                                                  $argname::$name, 
                                                                  args...) where {T <: Jaynes.Address, K}
            Jaynes.haskey(ctx.target, addr) || return Jaynes.get_value(Jaynes.get_sub(ctx.call, addr))
            s = Jaynes.getindex(ctx.call, addr)
            Jaynes.increment!(ctx, logpdf($argname, args..., s))
            return s
        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
