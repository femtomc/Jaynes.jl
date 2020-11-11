# ------------ Generate comilation context ------------ #

mutable struct GenerateContext{J <: CompilationOptions,
                               T <: AddressMap, 
                               K <: AddressMap, 
                               P <: AddressMap} <: ExecutionContext
    tr::T
    target::K
    weight::Float64
    score::Float64
    visited::Visitor
    params::P
    GenerateContext{J}(tr::T, target::K, weight::Float64, score::Float64, visited::Visitor, params::P) where {J, T, K, P} = new{J, T, K, P}(tr, target, weight, score, visited, params)
end

function Generate(::J, tr::AddressMap, target::AddressMap, params::AddressMap) where J <: CompilationOptions
    GenerateContext{J}(tr, 
                       target,
                       0.0,
                       0.0,
                       Visitor(),
                       params)
end

# ------------ Propose compilation context ------------ #

mutable struct ProposeContext{J <: CompilationOptions,
                              T <: AddressMap, 
                              P <: AddressMap} <: ExecutionContext
    map::T
    score::Float64
    visited::Visitor
    params::P
    ProposeContext{J}(tr::T, score::Float64, visited::Visitor, params::P) where {J, T, P} = new{J, T, P}(tr, score, visited, params)
end

function Propose(opt::J, tr, params) where J
    ProposeContext{J}(tr, 
                      0.0, 
                      Visitor(), 
                      params)
end

# ------------ Regenerate compilation context ------------ #

mutable struct RegenerateContext{J <: CompilationOptions,
                                 C <: AddressMap,
                                 T <: AddressMap, 
                                 K <: AddressMap,
                                 D <: AddressMap,
                                 P <: AddressMap} <: ExecutionContext
    prev::C
    tr::T
    target::K
    weight::Float64
    score::Float64
    discard::D
    visited::Visitor
    params::P
    RegenerateContext{J}(cl::C, tr::T, target::K, weight::Float64, score::Float64, discard::D, visited::Visitor, params::P) where {J, C, T, K, D, P} = new{J, C, T, K, D, P}(cl, tr, target, weight, score, discard, visited, params)
end

function Regenerate(::J, target::K, ps, cl::C, tr, discard) where {J <: CompilationOptions, K <: AddressMap, C <: CallSite}
    RegenerateContext{J}(cl, 
                         tr,
                         target, 
                         0.0, 
                         0.0, 
                         discard,
                         Visitor(), 
                         Empty())
end

# ------------ Simulate compilation context ------------ #

mutable struct SimulateContext{J <: CompilationOptions,
                               T <: AddressMap, 
                               P <: AddressMap} <: ExecutionContext
    tr::T
    score::Float64
    visited::Visitor
    params::P
    SimulateContext{J}(tr::T, score::Float64, visited::Visitor, params::P) where {J, T, P} = new{J, T, P}(tr, score, visited, params)
end

function Simulate(opt::J, tr::T, params) where {J, T}
    SimulateContext{J}(tr,
                       0.0, 
                       Visitor(), 
                       params)
end

# ------------ Update compilation context ------------ #

mutable struct UpdateContext{J <: CompilationOptions,
                             C <: CallSite, 
                             T <: AddressMap,
                             K <: AddressMap, 
                             D <: AddressMap,
                             P <: AddressMap} <: ExecutionContext
    prev::C
    tr::T
    target::K
    weight::Float64
    score::Float64
    discard::D
    visited::Visitor
    params::P
    UpdateContext{J}(cl::C, tr::T, target::K, weight, score, discard::D, vs::Visitor, params::P) where {J, C, T, K, D, P} = new{J, C, T, K, D, P}(cl, tr, target, weight, score, discard, vs, params)
end

function Update(::J, select::K, ps::P, cl::CL, tr, discard) where {J <: CompilationOptions, K <: AddressMap, P <: AddressMap, CL <: CallSite}
    UpdateContext{J}(cl, 
                     tr,
                     select, 
                     0.0, 
                     0.0, 
                     discard,
                     Visitor(), 
                     ps)
end

# ------------ Forward mode AD compilation context ------------ #

# Support for forward mode automatic differentiation.
mutable struct ForwardModeContext{J <: CompilationOptions,
                                  T <: Tuple,
                                  C <: AddressMap,
                                  D,
                                  P <: AddressMap} <: ExecutionContext
    target::T
    map::C
    weight::D
    visited::Visitor
    params::P
    ForwardModeContext{J}(target::T, map::C, weight::D, visited::Visitor, params::P) where {J, T, C, D, P} = new{J, T, C, D, P}(target, map, weight, visited, params)
end

function ForwardMode(addr, params, cl, weight)
    ForwardModeContext{DefaultPipeline}(addr, 
                                        cl, 
                                        weight, 
                                        Visitor(), 
                                        params)
end

# ------------ Assess compilation context ------------ #

mutable struct AssessContext{J <: CompilationOptions,
                             M <: AddressMap,
                             P <: AddressMap} <: ExecutionContext
    target::M
    weight::Float64
    visited::Visitor
    params::P
    AssessContext{J}(target::M, weight::Float64, visited::Visitor, params::P) where {J, M, P} = new{J, M, P}(target, weight, visited, params)
end

function Assess(opt::J, obs::AddressMap, params) where J
    AssessContext{J}(obs, 
                     0.0, 
                     Visitor(),
                     params)
end

# ------------ Backpropagation compilation contexts ------------ #

abstract type BackpropagationContext{J} <: ExecutionContext end

# Learnable parameters
mutable struct ParameterBackpropagateContext{J <: CompilationOptions,
                                             T <: CallSite, 
                                             S <: AddressMap,
                                             P <: AddressMap} <: BackpropagationContext{J}
    call::T
    weight::Float64
    fillables::S
    initial_params::P
    params::Store
    param_grads::Gradients
    ParameterBackpropagateContext{J}(call::T, weight::Float64, fillables::S, initial_params::P, params::Store, param_grads::Gradients) where {J, T, S, P} = new{J, T, S, P}(call, weight, fillables, initial_params, params, param_grads)
end

function ParameterBackpropagate(call::T, sel::S, init, params, param_grads::Gradients) where {T <: CallSite, S <: AddressMap, K <: Target}
    ParameterBackpropagateContext{DefaultPipeline}(call, 
                                                  0.0, 
                                                  sel, 
                                                  init, 
                                                  params, 
                                                  param_grads)
end

# Choice sites
mutable struct ChoiceBackpropagateContext{J <: CompilationOptions,
                                          T <: CallSite, 
                                          S <: AddressMap, 
                                          P <: AddressMap, 
                                          K <: Target} <: BackpropagationContext{J}
    call::T
    weight::Float64
    fillables::S
    initial_params::P
    choices::Store
    choice_grads::Gradients
    target::K
    ChoiceBackpropagateContext{J}(call::T, weight::Float64, fillables::S, initial_params::P, choices::Store, choice_grads::Gradients, target::K) where {J, T, S, P, K} = new{J, T, S, P, K}(call, weight, fillables, initial_params, choices, choice_grads, target)
end

function ChoiceBackpropagate(call::T, fillables::S, init, choice_store, choice_grads, sel::K) where {T <: CallSite, S <: AddressMap, K <: Target}
    ChoiceBackpropagateContext{DefaultPipeline}(call, 
                                                0.0, 
                                                fillables, 
                                                init, 
                                                choice_store,
                                                choice_grads,
                                                sel)
end


