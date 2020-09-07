const Δ = (v, d) -> Diffed(v, d)

DiffDefaults() = Multi(DiffPrimitives(), Mjolnir.Defaults())

partial_cleanup!(ir) = ir |> Mjolnir.inline_consts! |> Mjolnir.partials! |> Mjolnir.ssa! |> Mjolnir.prune! |> IRTools.renumber

function trace(P, Ts...)
    tr = Mjolnir.Trace(P)
    try
        argnames = [argument!(tr.ir, T) for T in Ts]
        for (T, x) in zip(Ts, argnames)
            T isa Union{Mjolnir.Partial, Mjolnir.Shape} && Mjolnir.node!(tr, T, x)
        end
        args = [T isa Const ? T.value : arg for (T, arg) in zip(Ts, argnames)]
        args, Ts = Mjolnir.replacement(P, args, Ts)
        if (T = Mjolnir.partial(tr.primitives, Ts...)) != nothing
            tr.total += 1
            Mjolnir.return!(tr.ir, push!(tr.ir, stmt(Expr(:call, args...), type = T)))
        else
            Mjolnir.return!(tr.ir, Mjolnir.tracecall!(tr, args, Ts...))
        end
        return partial_cleanup!(tr.ir)
    catch e
        throw(Mjolnir.TraceError(e, tr.stack))
    end
end

_propagate(a...) = trace(DiffDefaults(), a...)

create_flip_diff(a::Type{Diffed{K, DV}}) where {K, DV} = DV

@generated _pushforward(F, As...) = begin
    As = map(As) do a
        create_flip_diff(a)
    end
    tr = _propagate(F, As...)
    prune!(tr)
end

pushforward(fn, args...) = fn(args...), _pushforward(fn, args...)
