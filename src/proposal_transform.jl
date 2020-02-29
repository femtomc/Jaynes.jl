@dynamo function (inf_comp::InferenceCompiler)(m...)
    ir = IR(m...)
    ir == nothing && return
    recurse!(ir)
    return ir
end
