# GPUify :)
GPU(x::Array{Float64, N}) where N = CuArray(x)

# Constants for defining SELU function.
const α = 1.6732632423543772848170429916717
const λ = 1.0507009873554804934193349852946

SeLU(x) = x > 0 ? λ * x : α * CUDAnative.exp.(x) .- α

# Glorot uniform
glorot_uniform(in::Int, out::Int) = (lim = sqrt(6 / (in + out)); rand(Uniform(-lim, lim), (out, in)))
glorot_uniform(in::Int) = (lim = sqrt(6 / (in + in)); rand(Uniform(-lim, lim), in))

# Deep layers begin.
abstract type Layer end

# Dense.
struct Dense{T, K} <: Layer
    W::T
    b::K
    σ
end

# Default is Glorot uniform initializer.
Dense(in::Int, out::Int) = (lim = sqrt(6 / (in + out)); Dense(glorot_uniform(in, out), zeros(out), SeLU))
GPU(l::Dense) = Dense(CuArray(l.W), CuArray(l.b), l.σ)
(l::Dense{T, K})(x) where {T <: Array{Float64}, K <: Array{Float64, 1}} = l.σ.(l.W * x .+ l.b)
(l::Dense{T, K})(x::CuArray) where {T <: CuArray, K <: CuArray} = l.σ.(l.W * x .+ l.b)

# Chain layers together.
struct Chain{T <: Tuple}
    layers::T
    Chain(xs...) = new{typeof(xs)}(xs)
end
GPU(c::Chain) = Chain(map(l -> GPU(l), c.layers)...)
apply(::Tuple{}, x) = x
apply(fs::Tuple, x) = apply(Base.tail(fs), first(fs)(x))
(c::Chain)(x) = apply(c.layers, x)

# Recurrent cell.
mutable struct RNNCell{F, A, V}
    σ::F
    Wi::A
    Wh::A
    b::V
    h::V
end
GPU(r::RNNCell) = RNNCell(r.σ, GPU(r.Wi), GPU(r.Wh), GPU(r.b), GPU(r.h))
RNNCell(in::Int, out::Int, σ=SeLU) = RNNCell(σ, glorot_uniform(in, out), glorot_uniform(out, out), glorot_uniform(out), zeros(out))
hidden(rnn::RNNCell) = rnn.h

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

# Recur handles state.
mutable struct Recur{T}
    cell::T
    init
    state
end
GPU(r::Recur{T}) where T = Recur(GPU(r.cell), GPU(r.init), GPU(r.state))
Recur(m, h = hidden(m)) = Recur(m, h, h)

function (m::Recur)(xs...)
    h, y = m.cell(m.state, xs...)
    m.state = h
    return y
end

# Now begins the inference compiler work. An appropriate name :)
struct TalkingHead{T}
    name::Symbol
    model::T
end
(h::TalkingHead{T})(x) where T = h.model(x)
GPU(h::TalkingHead{T}) where T = TalkingHead(h.name, GPU(h.model))

# The type constraint forces you to use a recurrent transition "spine".
mutable struct InferenceCompiler{K <: Recur}
    encoding_heads::Dict{Symbol, TalkingHead}
    transition::K
    decoding_heads::Dict{Symbol, TalkingHead}
end
InferenceCompiler(rnn::T) where T <: Recur = InferenceCompiler(Dict{Symbol, TalkingHead}(), rnn, Dict{Symbol,TalkingHead}())

# Specifications setup an easy interface to communicate to the spine + heads.
abstract type Specification end

struct Post{T} <: Specification
    name::Symbol
    data::T
end
GPU(p::Post{T}) where T = Post(p.name, GPU(p.data))

struct Request{T} <: Specification
    name::Symbol
    data::T
end
GPU(r::Request{T}) where T = Request(r.name, GPU(r.data))

# create_head will spawn new heads if you see a spec which you haven't seen before. The created head has a dense model by default now.
function create_head(x::Post{T}, inf_comp::InferenceCompiler) where T
    name = x.name
    data = x.data
    flattened = map(i -> data[i], eachindex(data))
    return GPU(TalkingHead(name, Dense(length(flattened), length(inf_comp.transition.cell.h))))
end

function create_head(x::Request{T}, inf_comp::InferenceCompiler) where T
    name = x.name
    data = x.data
    flattened = map(i -> data[i], eachindex(data))
    return GPU(TalkingHead(name, Dense(length(inf_comp.transition.cell.h), length(flattened))))
end

# Transition.
function transition(z, inf_comp)
    return inf_comp.transition(z)
end

# Encode if head is available, else create head and then encode.
function send!(obs::Post{K}, inf_comp::InferenceCompiler) where K
    sym = obs.name
    if sym in keys(inf_comp.encoding_heads)
        data = obs.data
        out = transition(inf_comp.encoding_heads[sym](data), inf_comp)
        return out
    else
        inf_comp.encoding_heads[sym] = create_head(obs, inf_comp)
        return send!(obs, inf_comp)
    end
end

# Decode if head is available, else create head and then decode.
function send!(req::Request{K}, inf_comp::InferenceCompiler) where K
    sym = req.name
    if sym in keys(inf_comp.decoding_heads)
        data = inf_comp.decoding_heads[sym](inf_comp.transition.state)
        return data
    else
        inf_comp.decoding_heads[sym] = create_head(req, inf_comp)
        return send!(req, inf_comp)
    end
end
