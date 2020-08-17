# Support for forward mode automatic differentiation

mutable struct ForwardModeContext{T <: Tuple,
                                  C <: AddressMap,
                                  P <: AddressMap} <: ExecutionContext
    addr::T
    target::C
    weight::Dual
    visited::Visitor
    params::P
end

function ForwardMode(addr, tg)
    ForwardModeContext(addr, tg, Dual(0.0, 0.0), Visitor(), Empty())
end

function ForwardMode(addr, tg, params)
    ForwardModeContext(addr, tg, Dual(0.0, 0.0), Visitor(), params)
end

# ------------ includes ------------ #

include("dynamic/forwardmode.jl")

# ------------ Documentation ------------ #
