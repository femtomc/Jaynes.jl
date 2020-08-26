# ------------ Conditional map ------------ #

mutable struct ConditionalMap{K} <: AddressMap{K}
    cond::AddressMap{<:K}
    branch::AddressMap{<:K}
    ConditionalMap{K}() where K = new{K}()
    ConditionalMap{K}(cond::AddressMap{<:K}, branch::AddressMap{<:K}) where K = new{K}(cond, branch)
end
ConditionalMap(cond::AddressMap{K}, branch::AddressMap{K}) where K = ConditionalMap{K}(cond, branch)
Zygote.@adjoint ConditionalMap(cond, branch) = ConditionalMap(cond, branch), ret_grad -> (nothing, nothing)

@inline shallow_iterator(cm::ConditionalMap) = ((:cond, cm.cond), (:branch, cm.brand))

@inline function get(cm::ConditionalMap{Value}, addr, fallback)
    haskey(cm, addr) || return fallback
    return getindex(cm, addr)
end
