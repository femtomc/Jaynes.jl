module Simple

include("../../src/Jaynes.jl")
using .Jaynes
using Distributions

function foo(y::Float64)
    y = rand(Normal(0.5, 3.0))
    return y
end

@primitive function logpdf(fn::typeof(foo), args::Tuple{Float64}, y::Float64)
    if y < 1.0
        log(1) 
    else
        -Inf
    end
end

function bar(z::Float64)
    y = rand(:y, foo, (z, ))
    return y
end

ctx = Generate(Trace())
ret = trace(ctx, bar, (0.3, ))
display(ctx.metadata.tr)

end # module
