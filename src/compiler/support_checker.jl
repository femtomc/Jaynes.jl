# ------------ Common support errors, checked statically. ------------ #

# Checks for duplicate symbols - passes if addresses are in different blocks.
# Expects untyped IR.
function check_duplicate_symbols(ir)
    keys = Set{Symbol}([])
    blks = map(blk -> IRTools.BasicBlock(blk), blocks(ir))
    addresses = Dict( blk => Symbol[] for blk in blks )
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
            addr in addresses[r] ? error("SupportError: duplicate address $(st.expr.args[2]) found in IR.") : push!(addresses[r], addr)
        end
    end
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
    for (v, k) in types
        foldr(==, k) || begin
            println("SupportError: base measure mismatch at address $v. Derived support information:")
            display(types)
            error("SupportError.")
        end
    end
end
