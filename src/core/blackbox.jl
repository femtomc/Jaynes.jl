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
            ret = $argname(args...)
            score = logpdf($argname, ret)
            ctx.metadata.tr.chm[addr] = Choice(ret, score)
            push!(ctx.metadata.visited, addr)
            return ret
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
                ret = ctx.metadata.constraints[addr]
                score = logpdf($argname, ret)
                ctx.metadata.tr.chm[addr] = Choice(ret, score)
                ctx.metadata.tr.score += score
                push!(ctx.metadata.visited, addr)
                return ret
            else
                ret = $argname(args...)
                score = logpdf($argname, ret)
                ctx.metadata.tr.chm[addr] = Choice(ret, score)
                push!(ctx.metadata.visited, addr)
                return ret
            end
        end
    end
    expr = MacroTools.prewalk(unblock ∘ rmlines, expr)
    esc(expr)
end
