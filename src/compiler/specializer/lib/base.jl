# ------------ Runtime ------------ #

(a::Type{Array{K, N}})(init, d::Diffed) where {K, N} = a(init, unwrap(d))

# ------------ Abstract ------------ #

@abstract DiffPrimitives convert(args...) = propagate(args...)
@abstract DiffPrimitives iterate(args...) = propagate(args...)
@abstract DiffPrimitives getfield(args...) = propagate(args...)
@abstract DiffPrimitives typeassert(args...) = propagate(args...)
@abstract DiffPrimitives sleep(args...) = propagate(args...)
@abstract DiffPrimitives collect(args...) = propagate(args...)
