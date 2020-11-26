module BurglaryModel

include("../src/Jaynes.jl")
using .Jaynes
using Gen

jmodel = @jaynes function burglary_model()::Nothing
    burglary ~ Bernoulli(0.05)
    burglary ? disabled ~ Bernoulli(0.1) : disabled = false
    !disabled ? alarm ~ Bernoulli(burglary ? 0.94 : 0.01) : alarm = false
    call ~ Bernoulli(alarm ? 0.70 : 0.05)
    return nothing
end (NoStatic)

display(jmodel)

# Inference.
chm = choicemap((:call, true))
trs, lnws, lmle = importance_sampling(jmodel, (), chm, 5000)
est = sum(map(zip(lnws, trs)) do (lnw, tr)
              (tr[:burglary] ? 1 : 0) * exp(lnw)
          end)
println(est)

end # module
