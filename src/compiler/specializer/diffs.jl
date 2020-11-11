# ------------ Diff system ------------ #

struct ScalarDiff{K} <: Diff
    diff::K
end

struct BoolDiff <: Diff
    new::Bool
end

struct Change <: Diff end

@inline tupletype(dfs::Diffed...) = Tuple{map(d -> valtype(d), dfs)...}

valtype(d::Diffed{V, DV}) where {V, DV} = V

function change_check(args)
    unwrapped = map(args) do a
        unwrap(a) <: Change
    end
    any(unwrapped) && return Change
    return NoChange
end

function propagate(args...)
    unwrapped = map(args) do a
        unwrap(a)
    end
    change_check(args)
end

struct DiffPrimitives end

include("lib/numeric.jl")
include("lib/distributions.jl")
include("lib/base.jl")
