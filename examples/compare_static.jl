module CompareStatic

include("../src/Jaynes.jl")
using .Jaynes
using Gen
using IRTools

@gen (static) function foo(prob::Float64)
    z1 = @trace(bernoulli(prob), :a)
    z2 = @trace(bernoulli(prob), :b)
    z3 = z1 || z2
    z4 = !z3
    return z4
end

Gen.@load_generated_functions()

jmodel = @jaynes function boo(prob::Float64)
    z1 = {:a} ~ Bernoulli(prob)
    z2 = {:b} ~ Bernoulli(prob)
    z3 = z1 || z2
    z4 = !z3
    return z4
end

ftr = simulate(foo, (0.5, ))
btr = simulate(jmodel, (0.5, ))

display(get_choices(ftr))
display(btr)

chm = choicemap((:a, false))
st = static([(:a, ) => false])

update(ftr, (0.5, ), (NoChange(), ), chm)
@time update(ftr, (0.5, ), (NoChange(), ), chm)

update(btr, (0.5, ), (NoChange(), ), st)
@time update(btr, (0.5, ), (NoChange(), ), st)

update(btr, (0.5, ), (NoChange(), ), chm)
@time update(btr, (0.5, ), (NoChange(), ), chm)

end # module
