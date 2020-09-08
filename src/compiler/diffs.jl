# ------------ Diff system ------------ #

struct ScalarDiff{K} <: Diff
    diff::K
end

struct BoolDiff <: Diff
    new::Bool
end

struct Change <: Diff end

# Define the algebra for propagation of diffs.
unwrap(::Type{K}) where K = K
unwrap(::Const{K}) where K = K
unwrap(::Partial{K}) where K = K
unwrap(::Mjolnir.Node{K}) where K = K

function change_check(args)
    unwrapped = map(args) do a
        unwrap(a) <: Change
    end
    all(unwrapped) && return Change
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
