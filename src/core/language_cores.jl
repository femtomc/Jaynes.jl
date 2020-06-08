Cassette.@context DomainCtx

abstract type LanguageCore end
struct Base <: LanguageCore end
struct Functional <: LanguageCore end

macro accrete!(P, ex)
end

macro corrode!(P, ex)
end

mutable struct Language{T <: LanguageCore} <: Meta
    core::T
end

function Cassette.overdub(ctx::DomainCtx{M}, fn::Function, args) where M <: Language{T <: Functional}
    ret = dispatch_if_allowed(ctx.metadata.core, fn, args)
end
