# ------------ Anywhere map ------------ #

struct AnywhereMap{K} <: Leaf{K}
    tree::Dict{Any, Leaf{<:K}}
end
