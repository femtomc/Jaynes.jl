# Support for forward mode automatic differentiation.

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

function forward(addr, params, cl::DynamicCallSite, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(cl.fn, cl.args...)
    ret, ctx.weight
end

function forward(addr, params, cl::VectorCallSite{typeof(plate)}, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(plate, cl.fn, cl.args...)
    ret, ctx.weight
end

function forward(addr, params, cl::VectorCallSite{typeof(markov)}, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(markov, cl.fn, cl.args...)
    ret, ctx.weight
end

# Utility function which returns a Dual number of the index matches the target.
@inline function context_getindex(ctx, am, addr)
    if (addr, ) == ctx.target
        Dual(getindex(am, addr), 1.0)
    else
        getindex(am, addr)
    end
end

# ------------ Convenience ------------ #

function get_choice_gradient(addr::T, cl::C) where {T <: Tuple, C <: CallSite}
    fn = seed -> begin
        ret, w = forward(addr, Empty(), cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    cl[addr], d.partials.values[1]
end

function get_choice_gradient(addr::T, ps::P, cl::C) where {P <: AddressMap, T <: Tuple, C <: CallSite}
    fn = seed -> begin
        ret, w = forward(addr, ps, cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    cl[addr], d.partials.values[1]
end

function get_learnable_gradient(addr::T, ps::P, cl::C) where {P <: AddressMap, T <: Tuple, C <: CallSite}
    fn = seed -> begin
        ret, w = forward(addr, ps, cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    ps[addr], d.partials.values[1]
end

# ------------ includes ------------ #

include("dynamic/forwardmode.jl")
include("plate/forwardmode.jl")

# ------------ Documentation ------------ #
