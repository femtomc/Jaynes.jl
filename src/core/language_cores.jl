Cassette.@context DomainCtx

abstract type LanguageCore end
struct BaseLang <: LanguageCore end

mutable struct Language{T <: LanguageCore} <: Meta
    core::T
    toplevel::Function
    args::Tuple
    Language(c::T) where T <: LanguageCore = new{T}(c)
end

macro corrode!(P, ex)
    if @capture(shortdef(ex), f_)
        expr = quote
            function Jaynes.prehook(ctx::DomainCtx{M}, fn::typeof($f), args) where M <: Language{<: $P}
                error("$(typeof(ctx.metadata.core).name)Error: $fn with $(typeof(args)) is disallowed in this language.")
            end
        end
        esc(expr)
    end
end

macro accrete!(P, ex)
    if @capture(shortdef(ex), f_)
        expr = quote
            function Jaynes.prehook(ctx::DomainCtx{M}, fn::typeof($f), args) where M <: Language{<: $P}
                nothing
            end
        end
        esc(expr)
    end
end

function interpret(ctx, fn::Function, args...)
    ctx.metadata.toplevel = fn
    ctx.metadata.args = args
    ret = Cassette.overdub(ctx, fn, args...)
    return ret
end
