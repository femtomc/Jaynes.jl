module Caching

using Cassette
using Cassette: recurse

Cassette.@context CacheCtx

mutable struct CacheMeta
    cache::IdDict
    cache_targets::Vector{DataType}
    nocacheable::Set{Symbol}
    cache_all::Bool
    CacheMeta() = new(IdDict(), DataType[], Set(names(Core.Intrinsics)), false)
end
CacheMeta(tgs::Vector{DataType}) = (cm = CacheMeta(); cm.cache_targets = tgs; cm)

function factorial(n::Int)
    if n == 1
        return 1
    else
        return n * factorial(n - 1)
    end
end

function Cassette.overdub(ctx::CacheCtx, fn::Function, args...)
    haskey(ctx.metadata.cache, (fn, args...)) && return ctx.metadata.cache[(fn, args...)]
    ret = recurse(ctx, fn, args...)
    (typeof(fn) in ctx.metadata.cache_targets || 
     ctx.metadata.cache_all && !(Symbol(fn) in ctx.metadata.nocacheable)) && begin
        ctx.metadata.cache[(fn, args...)]  = ret
    end
    return ret
end

cm = CacheMeta([typeof(factorial)])
cm.cache_all = true
ctx = CacheCtx(metadata = cm)
ret = Cassette.overdub(ctx, factorial, 10)

println(cm)

end # module
