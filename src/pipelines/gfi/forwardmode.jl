# ------------ Staging ------------ #

@dynamo function (fx::ForwardModeContext{J})(a...) where J
    ir = IR(a...)
    ir == nothing && return
    ir = staged_pipeline(ir, ForwardModeContext{J})
    ir
end

# Base fixes.
(fx::ForwardModeContext)(::typeof(Core._apply_iterate), f, c::typeof(trace), args...) = fx(c, flatten(args)...)
function (fx::ForwardModeContext)(::typeof(Base.collect), generator::Base.Generator)
    map(generator.iter) do i
        fx(generator.f, i)
    end
end

# Utility function which returns a Dual number of the index matches the target.
@inline function context_getindex(ctx, am, addr)
    if (addr, ) == ctx.target
        Dual(getindex(am, addr), 1.0)
    else
        getindex(am, addr)
    end
end

function forward(addr, params, cl::DynamicCallSite, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(cl.fn, cl.args...)
    ret, ctx.weight
end

# ------------ Choice sites ------------ #

@inline function (ctx::ForwardModeContext)(call::typeof(trace), 
                                           addr::A, 
                                           d::Distribution{K}) where {A <: Address, K}
    visit!(ctx, addr)
    v = context_getindex(ctx, ctx.map, addr)
    ctx.weight += logpdf(d, v)
    v
end

# ------------ Learnable ------------ #

@inline function (ctx::ForwardModeContext)(fn::typeof(learnable), addr::Address)
    visit!(ctx, addr)
    haskey(ctx.params, addr) && return context_getindex(ctx, ctx.params, addr)
    error("Parameter not provided at address $addr.")
end

# ------------ Call sites ------------ #

@inline function (ctx::ForwardModeContext)(c::typeof(trace),
                                           addr::A,
                                           call::Function,
                                           args...) where A <: Address
    visit!(ctx, addr)
    ps = get_sub(ctx.params, addr)
    ret, w = forward(ctx.target[2 : end], ps, get_sub(ctx.map, addr), Dual(1.0, 0.0))
    ctx.weight += w
    return ret
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


function get_choice_gradient(addr::T, cl::DynamicCallSite) where T <: Tuple
    fn = seed -> begin
        ret, w = forward(addr, Empty(), cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    cl[addr], d.partials.values[1]
end

function get_choice_gradient(addr::T, ps::P, cl::DynamicCallSite) where {P <: AddressMap, T <: Tuple}
    fn = seed -> begin
        ret, w = forward(addr, ps, cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    cl[addr], d.partials.values[1]
end

function get_learnable_gradient(addr::T, ps::P, cl::DynamicCallSite) where {P <: AddressMap, T <: Tuple}
    fn = seed -> begin
        ret, w = forward(addr, ps, cl, seed)
        w
    end
    d = fn(Dual(1.0, 0.0))
    ps[addr], d.partials.values[1]
end

# ------------ Documentation ------------ #
