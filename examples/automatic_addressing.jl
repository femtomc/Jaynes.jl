module AutoAddressing2

include("../src/Jaynes.jl")
using .Jaynes

# Simple test.
foo = () -> begin
    y = rand(Normal(1.0, 3.0))
    return y
end

ret, cl = simulate(foo)
display(cl.trace)

# More complex test.
foo = () -> begin
    y = 0
    while y < 100
        y += rand(Normal(1.0, 3.0))
    end
    return y
end

ret, cl = simulate(foo)
display(cl.trace)

# Nested calls.
bar = () -> rand(Normal(5.0, 3.0))
foo = () -> begin
    rand(Bernoulli(0.5)) ? bar() : rand(Normal(10.0, 3.0))
end

ret, cl = simulate(foo)
display(cl.trace)

end # module
