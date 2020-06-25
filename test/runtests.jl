module TestJaynes

using Test

include("../src/Jaynes.jl")
using .Jaynes
using Distributions

@info "Testing the execution contexts functionality. This is core functionality."
include("contexts.jl")
@info "Testing the particle filtering functionality. This is inference functionality."
include("particle_filter.jl")
@info "Testing the importance sampling functionality. This is inference functionality."
include("importance_sampling.jl")
@info "Testing the Metropolis-Hastings functionality. This is inference functionality, a kernel for MCMC."
include("metropolis_hastings.jl")
@info "Testing the selection query language. This is core functionality."
include("selection_query_language.jl")

end #module
