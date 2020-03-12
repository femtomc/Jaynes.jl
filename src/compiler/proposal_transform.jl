@dynamo function (inf_comp::InferenceCompiler)(m...)
    ir = IR(m...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

function (inf_comp::InferenceCompiler)(call::typeof(Normal), params::K) where {T <: Distribution, K <: Array{Float64, 1}}
    name = gensym()
    params = send!(GPU(Request(name, Shape(2))), inf_comp)
    println(params)
    return call(name, params)
end

function (inf_comp::InferenceCompiler)(::typeof(rand), a::T) where {T <: Distribution}
    params = a.params
    result = rand(a)
    logprob = logpdf(a, result)
    add_trace!(inf_comp, a.name, Dict([:params => params, :result => result, :logpdf => logprob]))
    send!(GPU(Post(a.name, [result])), inf_comp)
    result
end

