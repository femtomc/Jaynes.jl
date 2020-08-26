# Support for forward mode automatic differentiation

mutable struct ForwardModeContext{T <: Tuple,
                                  C <: AddressMap,
                                  D,
                                  P <: AddressMap} <: ExecutionContext
    target::T
    map::C
    weight::D
    visited::Visitor
    params::P
end

function ForwardMode(addr, cl, weight)
    ForwardModeContext(addr, cl, weight, Visitor(), Empty())
end

function ForwardMode(addr, params, cl, weight)
    ForwardModeContext(addr, cl, weight, Visitor(), params)
end

function forward(addr, params, cl, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(cl.fn, cl.args...)
    ret, ctx.weight
end

@inline function context_getindex(ctx, am, addr)
    if (addr, ) == ctx.target
        Zygote.ForwardDiff.Dual(getindex(am, addr), 1.0)
    else
        getindex(am, addr)
    end
end

# ------------ includes ------------ #

include("dynamic/forwardmode.jl")

# ------------ Documentation ------------ #
