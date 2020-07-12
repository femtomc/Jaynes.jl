import Base.length
Base.length(ps::Particles) = length(ps.calls)

function initialize_filter(fn::Function, 
                           args::Tuple,
                           observations::ConstrainedHierarchicalSelection,
                           num_particles::Int)
    ps, lnw = importance_sampling(fn, args; observations = observations, num_samples = num_particles)
    return ps
end

function filter_step!(ps::Particles,
                      new_args::Tuple,
                      observations::ConstrainedHierarchicalSelection)
    num_particles = length(ps)
    for i in 1:num_particles
        ret, u_call, uw, retdiff, d = update(observations, ps.calls[i], new_args...)
        ps.calls[i].args = new_args
        ps.calls[i].ret = ret
        ps.lws[i] += uw
    end
    ltw = lse(ps.lws)
    ps.lmle = ltw - log(num_particles)
end

function filter_step!(ps::Particles,
                      new_args::Tuple,
                      proposal::Function,
                      proposal_args::Tuple,
                      observations::ConstrainedHierarchicalSelection)
    num_particles = length(ps)
    for i in 1:num_particles
        _, p_cl, p_w = propose(proposal, ps.calls[i], proposal_args...)
        sel = selection(p_cl)
        merge!(sel, observations)
        ret, u_cl, u_w, retdiff, d = update(sel, ps.calls[i], new_args...)
        ps.calls[i].args = new_args
        ps.calls[i].ret = ret
        ps.lws[i] += u_w - p_w
    end
    ltw = lse(ps.lws)
    ps.lmle = ltw - log(num_particles)
end

# Resample from existing set of particles, mutating the original set.
function resample!(ps::Particles)
    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_particles)
    lmle = ltw - log(num_particles)
    calls = map(selections) do ind
        ps.calls[ind]
    end
    ps.calls = calls
    ps.lws = zeros(num_particles)
end

function resample!(ps::Particles,
                   num::Int)
    num_particles = length(ps.calls)
    ltw, lnw = nw(ps.lws)
    weights = exp.(lnw)
    selections = rand(Categorical(weights/sum(weights)), num_particles)
    lmle = ltw - log(num_particles)
    calls = map(selections) do ind
        ps.calls[ind]
    end
    calls = rand(calls, num)
    ps.calls = calls
    ps.lws = zeros(length(ps.calls))
end

# ------------ Documentation ------------ #

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
