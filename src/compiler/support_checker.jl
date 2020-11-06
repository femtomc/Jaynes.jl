# ------------ Common support errors, checked statically. ------------ #

abstract type SupportException <: Exception end

struct MeasureMismatch <: SupportException
    types::Dict
    violations::Set{Symbol}
end
function Base.showerror(io::IO, e::MeasureMismatch)
    println("(SupportException) Base measure mismatch.")
    for (k, v) in e.types
        println(io, "$k => $(map(l -> pretty(l), v))")
    end
    println(io, "Violations: $(e.violations)")
    print(io, "\u001b[32mFix: ensure that base measures match for addresses shared across branches in your model.")
end

struct DuplicateAddresses <: SupportException
    violations::Set{Symbol}
end
function Base.showerror(io::IO, e::DuplicateAddresses)
    println("(SupportException) Duplicate addresses in same block.")
    println(io, "Violations: $(e.violations)")
    print(io, "\u001b[32mFix: ensure that all addresses in any straight line block are unique.")
end

# Checks for duplicate symbols - passes if addresses are in different blocks.
# Expects untyped IR.
function check_duplicate_symbols(ir)
    keys = Set{Symbol}([])
    blks = map(blk -> IRTools.BasicBlock(blk), blocks(ir))
    addresses = Dict( blk => Symbol[] for blk in blks )
    de = DuplicateAddresses(Set{Symbol}([]))
    for (v, st) in ir
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        unwrap(st.expr.args[1]) == :rand || continue
        st.expr.args[2] isa QuoteNode || continue
        addr = st.expr.args[2].value
        relevant = filter(blks) do blk
            st in blk.stmts
        end
        for r in relevant
            addr in addresses[r] ? push!(de.violations, addr) : push!(addresses[r], addr)
        end
    end
    isempty(de.violations) || throw(de)
end

# Checks that addresses across blocks share the same base measure. 
# Assumes that the input IR has been inferred (using e.g. the trace type inference).
function check_branch_support(tr)
    types = Dict()
    for (v, st) in tr
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        st.expr.args[1] == rand || continue
        st.expr.args[2] isa QuoteNode || continue
        addr = st.expr.args[2].value
        haskey(types, addr) ? push!(types[addr], supertype(st.type)) : types[addr] = Any[supertype(st.type)]
    end
    se = MeasureMismatch(types, Set{Symbol}([]))
    for (addr, supports) in types
        foldr(==, supports) || begin
            push!(se.violations, addr)
        end
    end
    isempty(se.violations) || throw(se)
end
