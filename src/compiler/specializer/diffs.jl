# ------------ Diff system ------------ #

struct ScalarDiff{K} <: Diff
    diff::K
end

struct BoolDiff <: Diff
    new::Bool
end

struct Change <: Diff end
struct Consumed <: Diff end

@inline tupletype(dfs::Diffed...) = Tuple{map(d -> valtype(d), dfs)...}

valtype(d::Diffed{V, DV}) where {V, DV} = V

# Diff propagation is basically a form of linear typing - Change types get "consumed" when they enter into a trace statement.

function propagate(args...)
    unwrapped = map(args) do a
        _lift(unwrap(a)) <: Change
    end
    any(unwrapped) && return Change
    return NoChange
end

function consume(args...)
    if any(args) do a
            a isa Change || _lift(a) <: Change
        end
        Consumed
    else
        NoChange
    end
end

struct DiffInterpreter <: InterpretationContext end

include("lib/numeric.jl")
include("lib/distributions.jl")
include("lib/base.jl")
