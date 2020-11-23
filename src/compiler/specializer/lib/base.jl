# ------------ Runtime ------------ #

(a::Type{Array{K, N}})(init, d::Diffed) where {K, N} = a(init, unwrap(d))
(f::Colon)(v::K, d::Diffed{K, NoChange}) where K = f(v, unwrap(d))
isless(v::K, d::Diffed{K, NoChange}) where K = isless(v, unwrap(d))
Base.Pair(v, d::Diffed) = Base.Pair(v, unwrap(d))
Base.Pair(d::Diffed, v) = Base.Pair(unwrap(d), v)

# ------------ Abstract ------------ #

@abstract DiffInterpreter convert(args...) = propagate(args...)
@abstract DiffInterpreter iterate(args...) = propagate(args...)
@abstract DiffInterpreter getfield(args...) = propagate(args...)
@abstract DiffInterpreter typeassert(args...) = propagate(args...)
@abstract DiffInterpreter sleep(args...) = propagate(args...)
@abstract DiffInterpreter collect(args...) = propagate(args...)
@abstract DiffInterpreter (::Colon)(args...) = propagate(args...)
@abstract DiffInterpreter Base.:(===)(args...) = propagate(args...)
@abstract DiffInterpreter (::Core.IntrinsicFunction)(args...) = propagate(args...)
@abstract DiffInterpreter (::Type{Pair})(args...) = propagate(args...)
