# Inference compilers always have talking heads.
struct TalkingHead{T}
    name::Symbol
    model::T
end
(h::TalkingHead{T})(x) where T = h.model(x) |> gpu

# The type constraint forces you to use a recurrent transition "spine".
mutable struct InferenceCompiler{K <: Recur}
    traces::PersistentHashMap{Symbol, Any}
    encoding_heads::Dict{Symbol, TalkingHead}
    transition::K
    decoding_heads::Dict{Symbol, TalkingHead}
end
InferenceCompiler(rnn::T) where T <: Recur = InferenceCompiler(PersistentHashMap{Symbol, Any}(), Dict{Symbol, TalkingHead}(), rnn, Dict{Symbol,TalkingHead}())
add_trace!(inf_comp::InferenceCompiler, addr::Symbol, record::Dict) = inf_comp.traces = assoc(inf_comp.traces, addr, record)
get_traces(inf_comp::InferenceCompiler) = inf_comp.traces

# Specifications setup an easy interface to communicate to the spine + heads.
abstract type Specification end

struct Post{T} <: Specification
    name::Symbol
    data::T
    Post{T}(name::Symbol, x::T) where T = new{T}(name, x |> gpu)
end

struct Request{T} <: Specification
    name::Symbol
    shape::T
    Request{T}(name::Symbol, shape::T) where T = new{T}(name, shape)
end

# create_head will spawn new heads if you see a spec which you haven't seen before. The created head has a dense model by default now.
function create_head(x::Post{T}, inf_comp::InferenceCompiler) where T
    name = x.name
    data = x.data
    flattened = reshape(data, (1,:))
    return TalkingHead(name, Dense(length(flattened), length(inf_comp.transition.cell.h)) |> gpu)
end

function create_head(x::Request{T}, inf_comp::InferenceCompiler) where T
    name = x.name
    shape = x.shape
    return TalkingHead(name, Dense(length(inf_comp.transition.cell.h), size(shape)) |> gpu)
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
