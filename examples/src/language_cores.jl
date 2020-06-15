module LanguageCores

include("../src/Jaynes.jl")
using .Jaynes
using Cassette

ctx = DomainCtx(metadata = Language(BaseLang()))

mutable struct Foo
    x::Float64
end

function foo(z::Float64)
    z = Foo(10.0)
    for i in 1:10
        Core.println(i)
    end
    x = 10
    if x < 15
        y = 20
    end
    y += 1
    z.x = 10.0
    return y
end

# Accepted!
ret = interpret(ctx, foo, 5.0)

@corrode! BaseLang setfield!
@corrode! BaseLang setproperty!
@corrode! BaseLang setindex!
@corrode! BaseLang Base.iterate
@accrete! BaseLang Base.iterate

# Rejected!
ret = interpret(ctx, foo, 5.0)


# Rejected!
ret = interpret(ctx, foo, 5.0)

end # module
