function var_ssa_map(f::Function, args...)
    lowered = @code_lowered f(args...)
    ir = @code_ir f(args...)
    vars = union(arguments(ir), keys(ir))
    return Dict{Symbol, Variable}(zip(lowered.slotnames, vars))
end
