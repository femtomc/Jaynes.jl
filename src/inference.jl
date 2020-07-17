# Utility structure for collections of samples.
mutable struct Particles{C}
    calls::Vector{C}
    lws::Vector{Float64}
    lmle::Float64
end

include("inference/is.jl")
include("inference/pf.jl")
include("inference/mh.jl")
include("inference/vi.jl")

# ------------ Documentation (IS) ------------ #

@doc(
"""
Samples from the model prior.
```julia
particles, normalized_weights = importance_sampling(model::Function, 
                                                    args::Tuple; 
                                                    observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                                                    num_samples::Int = 5000)
```
Samples from a programmer-provided proposal function.
```julia
particles, normalized_weights = importance_sampling(model::Function, 
                                                    args::Tuple, 
                                                    proposal::Function, 
                                                    proposal_args::Tuple; 
                                                    observations::ConstrainedSelection = ConstrainedAnywhereSelection(), 
                                                    num_samples::Int = 5000)
```

Run importance sampling on the posterior over unconstrained addresses and values. Returns an instance of `Particles` and normalized weights.
""", importance_sampling)

# ------------ Documentation (PF) ------------ #

@doc(
"""
```julia
particles = initialize_filter(fn::Function, 
                              args::Tuple,
                              observations::ConstrainedHierarchicalSelection,
                              num_particles::Int)
```
Instantiate a set of particles using a call to `importance_sampling`.
""", initialize_filter)

@doc(
"""
```julia
filter_step!(ps::Particles,
             new_args::Tuple,
             observations::ConstrainedHierarchicalSelection)
```
Perform a single filter step from an instance `ps` of `Particles`, applying the constraints specified by `observations`.

```julia
filter_step!(ps::Particles,
             new_args::Tuple,
             proposal::Function,
             proposal_args::Tuple,
             observations::ConstrainedHierarchicalSelection)
```
Perform a single filter step using a custom proposal function, applying the constraints specified by `observations`.
""", filter_step!)

@doc(
"""
```julia
resample!(ps::Particles)
resample!(ps::Particles, num::Int)
```
Resample from an existing instance of `Particles` by mutation in place.
""", resample!)

# ------------ Documentation (MH) ------------ #

@doc(
"""
```julia
call, accepted, metropolis_hastings(call::HierarchicalCallSite,
                                    sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using the prior at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.

```julia
call, accepted = metropolis_hastings(call::HierarchicalCallSite,
                                     proposal::Function,
                                     proposal_args::Tuple,
                                     sel::UnconstrainedSelection)
```

Perform a Metropolis-Hastings step by proposing new choices using a custom proposal at addressed specified by `sel`. Returns a call site, as well as a Boolean value `accepted` to indicate if the proposal was accepted or rejected.
""", metropolis_hastings)


# ------------ Documentation (VI) ------------ #

@doc(
"""
```julia
params, elbows, call_sites =  advi(sel::K,
                                   v_mod::Function,
                                   v_args::Tuple,
                                   mod::Function,
                                   args::Tuple;
                                   opt = ADAM(),
                                   iters = 1000, 
                                   gs_samples = 100) where K <: ConstrainedSelection
```
Given a selection `sel`, perform _automatic-differentiation variational inference_ with a proposal model `v_mod`. The result is a new set of trained parameters `params` for the variational model, the history of ELBO estimates `elbows`, and the call sites `calls` produced by the gradient estimator computation.
""", advi)
