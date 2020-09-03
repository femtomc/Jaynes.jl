const Δ = (v, d) -> Diffed(v, d)

DiffDefaults() = Multi(DiffPrimitives(), Mjolnir.Defaults())

_propagate(a...) = Mjolnir.trace(DiffDefaults(), a...)

@generated _forward(a...) = begin
    tr = _propagate(a...)
    tr
end

pushforward(fn, args...) = fn(args...), _forward(fn, args...)
