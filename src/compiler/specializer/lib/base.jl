# ------------ Runtime ------------ #

(a::Type{Array{K, N}})(init, d::Diffed) where {K, N} = a(init, unwrap(d))
(f::Colon)(v::K, d::Diffed{K, NoChange}) where K = f(v, unwrap(d))
isless(v::K, d::Diffed{K, NoChange}) where K = isless(v, unwrap(d))
Base.Pair(v, d::Diffed) = Base.Pair(v, unwrap(d))
Base.Pair(d::Diffed, v) = Base.Pair(unwrap(d), v)

# ------------ Abstract ------------ #

@abstract DiffPrimitives convert(args...) = propagate(args...)
@abstract DiffPrimitives iterate(args...) = propagate(args...)
@abstract DiffPrimitives getfield(args...) = propagate(args...)
@abstract DiffPrimitives typeassert(args...) = propagate(args...)
@abstract DiffPrimitives sleep(args...) = propagate(args...)
@abstract DiffPrimitives collect(args...) = propagate(args...)
