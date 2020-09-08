@abstract DiffPrimitives convert(to, from) = propagate(to, from)
@abstract DiffPrimitives iterate(iter, g) = propagate(iter, g)
@abstract DiffPrimitives getfield(v, b) = propagate(v, b)
