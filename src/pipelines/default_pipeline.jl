# -------- Default pipeline for model instantiation and each interface -------- #

# Deals with compilation at "model compile" time. 
# This occurs when the model structure is first constructed.
function instantiation_pipeline(fn::Function, arg_types::NTuple{N, Type}, ret_type::Type{R}, ::J) where {N, R, J <: DefaultCompilationOptions}
    opt = extract_options(J)
    ir = lower_to_ir(fn, arg_types...)
    opt.H && begin
        detect_kernels(fn, arg_types...)
        detect_switches(fn, arg_types...)
        detect_dynamic_addresses(fn, arg_types...)
    end
    println(" ________________________\n")
    opt.S ? tt = support_checker(TraceTypingInterpreter(), fn, arg_types...) : tt = missing
    println("\u001b[34m\e[1m   Trace type:\u001b[0m $tt")
    println("\u001b[34m\e[1m   Return type:\u001b[0m $R")
    println("\n\e[1m Finished compiling \e[4m$fn\u001b[0m.")
    println(" ________________________\n")
    tt, ir
end

# These pipelines operate at generated function expansion time (aka JIT/"inference runtime").

# DoesNotCare handles simulate, generate, propose, assess, backpropagate, etc - contexts which don't care about incremental computing with Diff types.
function staged_pipeline(ir, ::Type{J}, ::Type{K}, ::Type{A}) where {J <: DefaultCompilationOptions, K, A <: DoesNotCare}
    opt = extract_options(J)
    opt.AA && automatic_addressing_transform!(ir)
    ir = recur(ir)
    ir
end

# DiffAware is a Union type which handles update and regenerate contexts.
function staged_pipeline(ir, ::Type{J}, ::Type{K}, ::Type{A}) where {J <: DefaultCompilationOptions, K, A <: DiffAware}

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
