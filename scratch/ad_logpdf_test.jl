module TestADLogPDFNormal

using Zygote
using Distributions
using DistributionsAD: Normal

d = Normal(0.0, 1.0)
grad = gradient(pt -> logpdf(d, pt), 5.0)
println(grad)

end # module
