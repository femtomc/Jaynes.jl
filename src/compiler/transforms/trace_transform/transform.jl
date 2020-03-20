@dynamo function (trace::Trace)(m...)
    ir = IR(m...)
    ir == nothing && return
    recurse!(ir)
    return ir
end

function (tr::Trace)(call::typeof(rand), dist::T) where T <: Distribution
    result = call(dist)
    score = logpdf(dist, result)
    add_choice!(tr, pop!(tr.addresses), dist, result, score)
    return result
end
