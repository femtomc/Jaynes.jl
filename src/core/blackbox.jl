# This interface allows external calls/libraries to interact with the probabilistic tracing system. It defines new contextual primitives for overdubs - it requires that the user provide a specified logpdf.

macro primitive(ex)
    @capture(shortdef(ex), (logpdf(argname_::name_, args__) = body_) | (logpdf(argname_::name_, args__) where {Ts__} = body_)) || error("PrimitiveError: defining new contextual primitive calls requires a logpdf definition for the call.")
    argname = gensym(argname)
    expr = quote
        $ex
        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: UnconstrainedGenerateMeta, 
                                                     T <: Address}
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end
            addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")
            sample = $argname(args...)
            score = logpdf($argname, args, sample)
            ctx.metadata.tr.chm[addr] = Choice(sample, score)
            push!(ctx.metadata.visited, addr)
            return sample
        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: GenerateMeta, 
                                                     T <: Address}
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end
            addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")
            if haskey(ctx.metadata.constraints, addr)
                sample = ctx.metadata.constraints[addr]
                score = logpdf($argname, args, sample)
                ctx.metadata.tr.chm[addr] = Choice(sample, score)
                ctx.metadata.tr.score += score
                push!(ctx.metadata.visited, addr)
                return sample
            else
                sample = $argname(args...)
                score = logpdf($argname, args, sample)
                ctx.metadata.tr.chm[addr] = Choice(sample, score)
                push!(ctx.metadata.visited, addr)
                return sample
            end
        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: ProposalMeta, 
                                                     T <: Address}
            # Check stack.
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end

            # Check for support errors.
            addr in ctx.metadata.visited && error("AddressError: each address within a rand call must be unique. Found duplicate $(addr).")

            sample = $argname(args...)
            score = logpdf($argname, args, sample)
            ctx.metadata.tr.chm[addr] = Choice(sample, score)
            ctx.metadata.tr.score += score
            push!(ctx.metadata.visited, addr)
            return sample

        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: RegenerateMeta, 
                                                     T <: Address}
            # Check stack.
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end

            # Check if in previous trace's choice map.
            in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
            in_prev_chm && begin
                prev = ctx.metadata.tr.chm[addr]
                prev_val = prev.val
                prev_score = prev.score
            end

            # Check if in selection in meta.
            selection = ctx.metadata.selection
            in_sel = addr in selection

            ret = $argname(args...)
            in_prev_chm && !in_sel && begin
                ret = prev_val
            end

            score = logpdf($argname, args, ret)
            in_prev_chm && !in_sel && begin
                ctx.metadata.tr.score += score - prev_score
            end
            ctx.metadata.tr.chm[addr] = Choice(ret, score)

            # Visited
            push!(ctx.metadata.visited, addr)
            ret
        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: UpdateMeta, 
                                                     T <: Address}
            # Check stack.
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end

            # Check if in previous trace's choice map.
            in_prev_chm = haskey(ctx.metadata.tr.chm, addr)
            in_prev_chm && begin
                prev = ctx.metadata.tr.chm[addr]
                prev_ret = prev.val
                prev_score = prev.score
            end

            # Check if in constraints.
            in_constraints = haskey(ctx.metadata.constraints, addr)

            # Ret.
            if in_constraints
                ret = ctx.metadata.constraints[addr]
                push!(ctx.metadata.constraints_visited, addr)
            elseif in_prev_chm
                ret = prev_ret
            else
                ret = $argname(args...)
            end

            # Update.
            score = logpdf($argname, args, ret)
            if in_prev_chm
                ctx.metadata.tr.score += score - prev_score
            elseif in_constraints
                ctx.metadata.tr.score += score
            end
            ctx.metadata.tr.chm[addr] = Choice(ret, score)

            # Visited.
            push!(ctx.metadata.visited, addr)
            return ret
        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                        call::typeof(rand), 
                                        addr::T, 
                                        $argname::$name,
                                        args) where {M <: ScoreMeta, 
                                                     T <: Address}
            # Check stack.
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end

            # Get val.
            val = ctx.metadata.tr.chm[addr].value
            ctx.metadata.tr.score += logpdf($argname, args, val)

            # Visited.
            push!(ctx.metadata.visited, addr)
            return val
        end

        @inline function Jaynes.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  $argname::$name,
                                  args) where {M <: UnconstrainedGradientMeta, 
                                               T <: Address}

            # Check stack.
            !isempty(ctx.metadata.stack) && begin
                push!(ctx.metadata.stack, addr)
                addr = foldr((x, y) -> x => y, ctx.metadata.stack)
                pop!(ctx.metadata.stack)
            end

            # Build dependency graph.
            passed_in = filter(args) do a 
                if a in keys(ctx.metadata.tracker)
                    k = ctx.metadata.tracker[a]
                    if haskey(ctx.metadata.parents, addr) && !(k in ctx.metadata.parents[addr])
                        push!(ctx.metadata.parents[addr], k)
                    else
                        ctx.metadata.parents[addr] = [k]
                    end
                    true
                else
                    false
                end
            end

            args = map(args) do a
                if haskey(ctx.metadata.tracker, a)
                    k = ctx.metadata.tracker[a]
                    if haskey(ctx.metadata.trainable, k)
                        return ctx.metadata.trainable[k][1]
                    else
                        a
                    end
                else
                    a
                end
            end

            passed_in = IdDict{Any, Address}(map(passed_in) do a
                                                 k = ctx.metadata.tracker[a]
                                                 a => k
                                             end)

            # Check trace for choice map.
            !haskey(ctx.metadata.tr.chm, addr) && error("UnconstrainedGradientMeta: toplevel function call has address space which does not match the training trace.")
            sample = ctx.metadata.tr.chm[addr].val
            ctx.metadata.tracker[sample] = addr

            # Gradients
            gs = Flux.gradient((s, a) -> (loss = -logpdf($argname, a, s);
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
                map(ctx.metadata.parents[addr]) do p
                    p in keys(ctx.metadata.trainable) && begin
                        if haskey(ctx.metadata.gradients, p)
                            push!(ctx.metadata.gradients[p], grads)

                        else
                            ctx.metadata.gradients[p] = [grads]
                        end
                    end
                end
            end

            push!(ctx.metadata.visited, addr)
            return sample
        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
