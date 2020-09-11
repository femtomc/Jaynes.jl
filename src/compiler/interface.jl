const Δ = (v, d) -> Diffed(v, d)

DiffDefaults() = Multi(DiffPrimitives(), Mjolnir.Defaults())

# A bunch of nice passes which clean up the IR after tracing. Other cleaning passes can be found in transforms.jl.
partial_cleanup!(ir) = ir |> Mjolnir.inline_consts! |> Mjolnir.partials! |> Mjolnir.ssa! |> Mjolnir.prune! |> IRTools.renumber

# This is a modified version of Mjolnir's trace which grabs the IR associated with the original svec of types defined by the user - but then replaces the argtypes with diff types and does type inference.
function trace(P, f, Dfs, Ts...)
    tr = Mjolnir.Trace(P)
    try
        argnames = [argument!(tr.ir, T) for T in (f, Dfs...)]
        for (T, x) in zip(Ts, argnames)
            T isa Union{Mjolnir.Partial, Mjolnir.Shape} && Mjolnir.node!(tr, T, x)
        end
        args = [T isa Const ? T.value : arg for (T, arg) in zip((f, Ts...), argnames)]
        args, Ts = Mjolnir.replacement(P, args, (f, Ts...))
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

# Convenient - run trace with DiffDefaults primitives.
_propagate(f, args, Dfs) = trace(DiffDefaults(), f, Dfs, args...)

function create_flip_diff(a::Type{Diffed{K, DV}}) where {K, DV}
    DV != NoChange && return Change
    NoChange
end

# These are not currently used at generated function expansion time.
@generated _pushforward(F, As...) = begin
    ir = IRTools.IR(IRTools.meta(Tuple{F, As...}))
    As = map(As) do a
        create_flip_diff(a)
    end
    tr = _propagate(F, As...)
    ir = prune(tr)
    strip_types!(ir)
    ir
end

pushforward(fn, args...) = fn(args...), _pushforward(fn, args...)
