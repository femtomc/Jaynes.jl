# Check literals.
_lit(v::Variable) = false
_lit(v::Type) = false
_lit(v) = true

# Lift literals to types.
_lift(t::Type) = t
_lift(t) = typeof(t)

# Lower a function signature directly to IR.
function lower_to_ir(call, argtypes...)
    sig = length(argtypes) == 1 && argtypes[1] == Tuple{} ? begin
        Tuple{typeof(call)}
    end : Tuple{typeof(call), argtypes...}
    m = meta(sig)
    ir = IR(m)
    return ir
end

# Returns the signature of the matching method, specializing types as necessary.
function signature(fn_type::Type, arg_types::Type...)
    meta = IRTools.meta(Tuple{fn_type, arg_types...})
    if isnothing(meta) return nothing end
    sig, typevars = unparameterize(meta.method.sig)
    sig_types = collect(sig.parameters[2:end])
    for (i, a) in enumerate(arg_types)
        if sig_types[i] isa TypeVar
            sig_types[i] = a
        else
            sig_types[i] = reparameterize(sig_types[i], typevars)
        end
    end
    return sig_types
end

# Unwraps UnionAll types, returning the innermost body with a list of TypeVars.
function unparameterize(@nospecialize(T::Type))
    vars = TypeVar[]
    while T isa UnionAll
        pushfirst!(vars, T.var)
        T = T.body
    end
    return T, vars
end

# Rewraps parametric types with the provided TypeVars.
function reparameterize(@nospecialize(T::Type), vars::Vector{TypeVar})
    if T isa UnionAll
        T, origvars = unparameterize(T)
        vars = unique!([origvars; vars])
    end
    parameters = Set(T.parameters)
    for v in vars
        if (v in parameters) T = UnionAll(v, T) end
    end
    return T
end

# Returns a Vector of Tuple{Vararg{Int}} representing all possible combinations of flow of control.
function get_control_flow_paths(blk::Int, v::Vector{Vector{Int}})
    isempty(v[blk]) && return [(blk, )]
    paths = []
    for tar in filter(b -> b > blk, v[blk]) # prevent looping
        append!(paths, map(get_control_flow_paths(tar, v)) do p
                    (blk, p...)
                end)
    end
    paths
end
function get_control_flow_paths(cfg::CFG)
    graph = cfg.graph
    paths = get_control_flow_paths(1, graph)
    paths
end
@inline get_control_flow_paths(ir::IR) = get_control_flow_paths(CFG(ir))

# Check for control flow in IR - if multiple basic blocks, transfer of control is present.
@inline control_flow_check(ir) = !(length(ir.blocks) > 1)
