Cassette.@context DomainCtx

abstract type LanguageCore end

struct BaseLang <: LanguageCore end
disallowed(::BaseLang, fn::Function, args::Tuple) = return false
allowed(::BaseLang, fn::Function, args::Tuple) = return true
disallowed(::BaseLang, fn::Function) = return false
allowed(::BaseLang, fn::Function) = return true

macro accrete!(P, ex)
    if @capture(shortdef(ex), f_)
        expr = quote
            import Jaynes.allowed
            Jaynes.allowed($P, $f) = return true
        end
        expr
    end
end

macro corrode!(P, ex)
    if @capture(shortdef(ex), f_)
        expr = quote
            import Jaynes.disallowed
            function Jaynes.disallowed(::$(P), ::typeof($f))
                return true
            end
        end
        esc(expr)
    end
end

mutable struct Language{T <: LanguageCore} <: Meta
    core::T
    toplevel::Function
    args::Tuple
    Language(c::T) where T <: LanguageCore = new{T}(c)
end

function Cassette.prehook(ctx::DomainCtx{M}, fn::Function, args...) where M <: Meta
    x = disallowed(ctx.metadata.core, fn)
    if x
        error("LanguageCoreError: $fn with $(typeof(args)) is disallowed.")
    end
end

function Cassette.overdub(ctx::DomainCtx{M}, fn::Function, args...) where M <: Meta
    (allowed(ctx.metadata.core, fn, args) || fn == ctx.metadata.toplevel) && begin
        canrecurse(ctx, fn, args...) && return recurse(ctx, fn, args...)
        return fn(args...)
    end
    error("LanguageCoreError: $fn with $(typeof(args)) not allowed.")
end

function interpret(ctx, fn::Function, args...)
    ctx.metadata.toplevel = fn
    ctx.metadata.args = args
    ret = Cassette.overdub(ctx, fn, args...)
    return ret
end
