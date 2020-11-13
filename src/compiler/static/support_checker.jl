# ------------ Common support errors, checked statically. ------------ #

abstract type SupportException <: Exception end

# Returns a Vector of Tuple{Vararg{Int}} representing all possible combinations of flow of control.
function get_control_flow_paths(blk::Int, v::Vector{Vector{Int}})
    isempty(v[blk]) && return [(blk, )]
    paths = []
    for tar in filter(b -> b != blk, v[blk]) # prevent looping
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

# ------------ Mismatch measures across flow of control paths ------------ #

struct MeasureMismatch <: SupportException
    types::Dict
    violations::Set{Symbol}
end
function Base.showerror(io::IO, e::MeasureMismatch)
    println("\u001b[31m(SupportException):\u001b[0m Base measure mismatch.")
    for (k, v) in e.types
        println(io, " $k => $(map(l -> pretty(l), v))")
    end
    println(io, " Violations: $(e.violations)")
    println(io, "\u001b[32mFix: ensure that base measures match for addresses shared across branches in your model.\u001b[0m")
end

# Checks that addresses across flow of control paths share the same base measure. 
# Assumes that the input IR has been inferred (using e.g. the trace type inference).
function check_branch_support(tr)
    types = Dict()
    for (v, st) in tr
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        st.expr.args[1] == trace || continue
        st.expr.args[2] isa QuoteNode || continue
        addr = st.expr.args[2].value
        if haskey(types, addr)
            st.type isa NamedTuple ? push!(types[addr], st.type) : push!(types[addr], supertype(st.type))
        else
            st.type isa NamedTuple ? Any[st.type] : Any[supertype(st.type)]
        end
    end
    se = MeasureMismatch(types, Set{Symbol}([]))
    for (addr, supports) in types
        length(supports) == 1 && continue
        foldr(==, supports) || begin
            push!(se.violations, addr)
        end
    end
    se
end

# ------------ Duplicate addresses across flow of control paths ------------ #

struct DuplicateAddresses <: SupportException
    violations::Set{Symbol}
    paths::Set
end
function Base.showerror(io::IO, e::DuplicateAddresses)
    println("\u001b[31m(SupportException):\u001b[0m Duplicate addresses along same flow of control path in model program.")
    println(io, " Violations: $(e.violations)")
    println(io, " On flow of control paths:")
    for p in e.paths
        println(io, " $p")
    end
    println(io, "\u001b[32mFix: ensure that all addresses in any execution path through the program are unique.\u001b[0m")
end

# Checks for duplicate symbols along each flow of control path.
# Expects untyped IR.
function check_duplicate_symbols(ir, paths)
    keys = Set{Symbol}([])
    blks = map(blk -> IRTools.BasicBlock(blk), blocks(ir))
    addresses = Dict( path => Symbol[] for path in paths )
    de = DuplicateAddresses(Set{Symbol}([]), Set([]))
    for (v, st) in ir
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        unwrap(st.expr.args[1]) == :trace || continue
        st.expr.args[2] isa QuoteNode || continue
        addr = st.expr.args[2].value

        # Fix: can be much more efficient.
        relevant = Iterators.filter(Iterators.enumerate(blks)) do (ind, blk)
            st in blk.stmts
        end

        # Filter paths, then check if already seen on path.
        for (ind, blk) in relevant
            for p in filter(p -> ind in p, paths)
                if addr in addresses[p]
                    push!(de.violations, addr)
                    push!(de.paths, p)
                else
                    push!(addresses[p], addr)
                end
            end
        end
    end
    de
end

# ------------ Pipeline ------------ #

function support_checker(fn, arg_types...)
    ir = lower_to_ir(fn, arg_types...)
    errs = Exception[]
    paths = get_control_flow_paths(ir)
    push!(errs, check_duplicate_symbols(ir, paths))
    tr = infer_support_types(fn, arg_types...)
    !(tr isa Missing) ? push!(errs, check_branch_support(tr)) : println("\u001b[33m (SupportChecker): model program could not be traced. Branch support checks cannot run.\u001b[0m")
    any(map(errs) do err
            if isempty(err.violations)
                false
            else
                Base.showerror(stdout, err)
                true
            end
        end) ? error("SupportError found.") : println("\u001b[32m ✓ (SupportChecker): no errors found. If branch support checks could not be run, this message indicates that your model program is free from duplicate addresses.\u001b[0m")
    !(tr isa Missing) && begin
        if !control_flow_check(tr)
            @info "Detected control flow in model IR. Static trace typing requires that control flow be extracted into combinators."
            return missing
        else
            try
                return trace_type(tr)
            catch e
                @info "Failed to compute trace type. Caught:\n$e.\n\nProceeding to compile with missing trace type."
                return missing
            end
        end
        return missing
    end
end
