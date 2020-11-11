# ------------ Default pass pipelines for each interface ------------ #

function pipeline(ir, ::Type{AssessContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function pipeline(ir, ::Type{UpdateContext{J}}, ::Type{K}) where {J <: DefaultCompilationOptions, K}
    
    # Get option type.
    opt = extract_options(J)

    # Equivalent to static DSL optimizations.
    if K <: DynamicMap || !control_flow_check(ir) || opt.Spec == :off

        # Release IR normally.
        opt.AA == :on && automatic_addressing_transform!(ir)
        ir = recur(ir)
        argument!(ir, at = 2) # Must include for "sneaky invoke" trick.
        ir = renumber(ir)
    else

        # Argument difference inference.
        tr = diff_inference(f, S.parameters, args)

        # Dynamic specialization transform.
        ir = optimization_pipeline(ir.meta, tr, get_address_schema(K))

        # Automatic addressing transform.
        opt.AA == :on && automatic_addressing_transform!(ir)
    end
    ir
end

function pipeline(ir, ::Type{RegenerateContext{J}}, ::Type{K}) where {J <: DefaultCompilationOptions, K}
   
    # Get option type.
    opt = extract_options(J)

    # Equivalent to static DSL optimizations.
    if K <: DynamicMap || !control_flow_check(ir) || opt.Spec == :off

        # Release IR normally.
        opt.AA == :on && automatic_addressing_transform!(ir)
        ir = recur(ir)
        argument!(ir, at = 2) # Must include for "sneaky invoke" trick.
        ir = renumber(ir)
    else

        # Argument difference inference.
        tr = diff_inference(f, S.parameters, args)

        # Dynamic specialization transform.
        ir = optimization_pipeline(ir.meta, tr, get_address_schema(K))

        # Automatic addressing transform.
        opt.AA == :on && automatic_addressing_transform!(ir)
    end
    ir
end

function pipeline(ir, ::Type{GenerateContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function pipeline(ir, ::Type{SimulateContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function pipeline(ir, ::Type{ProposeContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function pipeline(ir, ::Type{ForwardModeContext{J}}) where J <: DefaultCompilationOptions
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

function pipeline(ir, ::Type{BackpropagationContext{J}}) where J
    opt = extract_options(J)
    opt.AA == :on && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end
