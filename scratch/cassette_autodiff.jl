module CassetteAutodiff

using Cassette

Cassette.@context J; # J is the notation for the function which generates pullbacks (the lambda terms you see below)

# Equivalent to gradient tape.
mutable struct ReverseTrace
    gradient_tape::Array{Function, 1}
    ReverseTrace() = new([])
end

function apply(arr::Array{Function, 1}, arg)
    arg = arg
    while length(arr) > 0
        func = pop!(arr)
        arg = func(arg)
    end
    return arg
end

gradient(tr::ReverseTrace) = apply(tr.gradient_tape, 1)

# Trace with context.
function Cassette.overdub(ctx::J, func::typeof(sin), arg::Float64)
    result = sin(arg)
    lambda = ȳ-> ȳ * cos(arg)
    push!(ctx.metadata.gradient_tape, lambda)
    return result
end

function Cassette.overdub(ctx::J, func::typeof(cos), arg::Float64)
    result = cos(arg)
    lambda = ȳ-> ȳ * -sin(arg)
    push!(ctx.metadata.gradient_tape, lambda)
    return result
end

# Test.
function foo(x::Float64)
    y = sin(x)
    z = cos(y)
    return z
end

tr = ReverseTrace()
Cassette.overdub(Cassette.disablehooks(J(metadata = tr)), foo, 5.0)
println(tr.gradient_tape)
println(gradient(tr))

end #module
