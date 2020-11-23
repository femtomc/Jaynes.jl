# Lift a value and a diff instance to a wrapped Diffed instance.
Δ(v, d) = Diffed(v, d)
Δ(v, d::Type) = Diffed(v, d())

# A bunch of nice passes which clean up the IR after tracing. 
# Other cleaning passes can be found in transforms.jl.
# This function _does not_ remove dead code (because this interferes with inference semantics).
partial_cleanup!(ir) = ir |> inline_consts! |> partials! |> ssa! |> prune! |> IRTools.renumber

# Convenience - run trace with DiffInterpreter primitives.
function _propagate(f, Dfs, args)
    ir = lower_to_ir(f, args...)
    typed_args = map(zip(IRTools.arguments(ir)[2 : end], Dfs)) do (a, t)
        a => t
    end
    env = Dict{Any, Any}(typed_args)
    env[IRTools.var(1)] = f
    tr = infer!(DiffInterpreter(), env, ir)
    tr
end

# Lift a runtime diff type to the simple indicator Change/NoChange types.
function create_flip_diff(a::Type{Diffed{K, DV}}) where {K, DV}
    DV != NoChange && return Change
    NoChange
end
