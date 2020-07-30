import Base.map

# Utility structure for collections of samples.
mutable struct Particles{C}
    calls::Vector{C}
    lws::Vector{Float64}
    lmle::Float64
end

map(fn::Function, ps::Particles) = map(fn, ps.calls)

include("inference/is.jl")
include("inference/pf.jl")
include("inference/mh.jl")
include("inference/vi.jl")
include("inference/hmc.jl")

const hmc = hamiltonian_monte_carlo
const mh = metropolis_hastings
const mh! = metropolis_hastings!
const is = importance_sampling
const advi = automatic_differentiation_variational_inference

# ------------ Documentation (IS) ------------ #

@doc(
"""
Samples from the model prior.
```julia
particles, normalized_weights = importance_sampling(observations::ConstrainedSelection,
                                                    num_samples::Int,
                                                    model::Function, 
                                                    args::Tuple)
```
Samples from a programmer-provided proposal function.
```julia
particles, normalized_weights = importance_sampling(observations::ConstrainedSelection,
                                                    num_samples::Int,
                                                    model::Function, 
                                                    args::Tuple, 
                                                    proposal::Function, 
                                                    proposal_args::Tuple)
```

Run importance sampling on the posterior over unconstrained addresses and values. Returns an instance of `Particles` and normalized weights.
""", importance_sampling)

# ------------ Documentation (PF) ------------ #

@doc(
"""
```julia
particles = initialize_filter(observations::ConstrainedHierarchicalSelection,
                              num_particles::Int,
                              fn::Function, 
                              args::Tuple)
```

Instantiate a set of particles using a call to `importance_sampling`.
""", initialize_filter)

@doc(
"""
```julia
filter_step!(observations::ConstrainedHierarchicalSelection,
             ps::Particles,
             new_args::Tuple)
```
Perform a single filter step from an instance `ps` of `Particles`, applying the constraints specified by `observations`.

```julia
filter_step!(observations::ConstrainedHierarchicalSelection,
             ps::Particles,
             new_args::Tuple,
             proposal::Function,
             proposal_args::Tuple)
```
Perform a single filter step using a custom proposal function, applying the constraints specified by `observations`.
""", filter_step!)

@doc(
"""
```julia
check_ess_resample!(ps::Particles)
```
Checks the effective sample size using `ess`, then resamples from an existing instance of `Particles` by mutation in place.
""", check_ess_resample!)

# ------------ Documentation (MH) ------------ #

@doc(
"""
```julia
call, accepted, metropolis_hastings(sel::UnconstrainedSelection,
                                    call::HierarchicalCallSite)
```

Perform a Metropolis-Hastings step by proposing new choices using the prior at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.

```julia
call, accepted = metropolis_hastings(sel::UnconstrainedSelection,
                                     call::HierarchicalCallSite,
                                     proposal::Function,
                                     proposal_args::Tuple)
```

Perform a Metropolis-Hastings step by proposing new choices using a custom proposal at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.
""", metropolis_hastings)


# ------------ Documentation (VI) ------------ #

@doc(
"""
```julia
params, elbows, call_sites =  advi(sel::K,
                                   iters::Int,
                                   v_mod::Function,
                                   v_args::Tuple,
                                   mod::Function,
                                   args::Tuple;
                                   opt = ADAM(),
                                   gs_samples = 100) where K <: ConstrainedSelection
```

Given a selection `sel`, perform _automatic-differentiation variational inference_ with a proposal model `v_mod`. The result is a new set of trained parameters `params` for the variational model, the history of ELBO estimates `elbows`, and the call sites `calls` produced by the gradient estimator computation.
""", advi)
