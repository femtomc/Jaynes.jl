# -------- Default pipeline for model instantiation and each interface -------- #

function instantiation_pipeline(fn::Function, arg_types::NTuple{N, Type}, ret_type::Type{R}, ::J) where {N, R, J <: DefaultCompilationOptions}
    opt = extract_options(J)
    ir = lower_to_ir(fn, arg_types...)
    opt.H && begin
        detect_kernels(fn, arg_types...)
        detect_dynamic_addresses(fn, arg_types...)
    end
    opt.S ? tt = support_checker(TraceTypingInterpreter(), fn, arg_types...) : tt = missing
    tt, ir
end

function staged_pipeline(ir, ::Type{AssessContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function staged_pipeline(ir, ::Type{UpdateContext{J}}, ::Type{K}) where {J <: DefaultCompilationOptions, K}

    # Get option type.
    opt = extract_options(J)

    # Equivalent to static DSL optimizations.
    if K <: DynamicMap || !control_flow_check(ir) || !opt.Sp

        # Release IR normally.
        opt.AA && automatic_addressing_transform!(ir)
        ir = recur(ir)
        argument!(ir, at = 2) # Must include for "sneaky invoke" trick.
        ir = renumber(ir)
    else

        # Argument difference inference.
        tr = diff_inference(f, S.parameters, args)

        # Dynamic specialization transform.
        ir = optimization_staged_pipeline(ir.meta, tr, get_address_schema(K))

        # Automatic addressing transform.
        opt.AA && automatic_addressing_transform!(ir)
    end
    ir
end

function staged_pipeline(ir, ::Type{RegenerateContext{J}}, ::Type{K}) where {J <: DefaultCompilationOptions, K}

    # Get option type.
    opt = extract_options(J)

    # Equivalent to static DSL optimizations.
    if K <: DynamicMap || !control_flow_check(ir) || !opt.Sp

        # Release IR normally.
        opt.AA && automatic_addressing_transform!(ir)
        ir = recur(ir)
        argument!(ir, at = 2) # Must include for "sneaky invoke" trick.
        ir = renumber(ir)
    else

        # Argument difference inference.
        tr = diff_inference(f, S.parameters, args)

        # Dynamic specialization transform.
        ir = optimization_staged_pipeline(ir.meta, tr, get_address_schema(K))

        # Automatic addressing transform.
        opt.AA && automatic_addressing_transform!(ir)
    end
    ir
end

function staged_pipeline(ir, ::Type{GenerateContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function staged_pipeline(ir, ::Type{SimulateContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function staged_pipeline(ir, ::Type{ProposeContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function staged_pipeline(ir, ::Type{ForwardModeContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function staged_pipeline(ir, ::Type{BackpropagationContext{J}}) where J
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end
