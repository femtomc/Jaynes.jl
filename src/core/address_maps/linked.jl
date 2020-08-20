# ------------ Linked map ------------ #

mutable struct LinkedMap{K} <: AddressMap{K}
    pointer::K
    next::AddressMap{<:K}
    LinkedMap{K}() where K = new{K}()
    LinkedMap(pointer::K) where K = new{K}(pointer)
    LinkedMap{K}(next::AddressMap{<:K}) where K = new{K}(next)
end
