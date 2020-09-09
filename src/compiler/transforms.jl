# Get all successors of a register.
function get_successors(v, ir)
    children = Vector{Variable}([])
    for (ch, st) in ir
        check = false
        MacroTools.postwalk(st.expr) do ex
            ex isa Variable && v == ex && push!(children, ch)
            ex
        end
    end
    children
end

# Get variable dependency graph from IR.
function dependencies(ir)
    d = Dict()
    map(keys(ir)) do v
        d[v] = get_successors(v, ir)
    end
    d
end

@inline check_change(st) = st.type == Change

# This pass is a dataflow analysis which determines what deterministic statements can be safely removed.
function no_change_prune(ir)
    d = dependencies(ir)
    for v in reverse(keys(ir))
    end
    ir
end

# This pass strips the types from all basic block arguments (and thus, branches).
function strip_types(ir)
    pr = IRTools.Pipe(ir)
    for (v, st) in pr
        pr[v] = IRTools.Statement(st, type = Any)
    end
    new = IRTools.finish(pr)
    for bb in IRTools.blocks(new)
        at = IRTools.argtypes(bb)
        for (i, a) in enumerate(at)
            at[i] = Any
        end
    end
    new
end

# This pass wraps rand calls.
function rand_wrapper(ir)
    pr = IRTools.Pipe(ir)
    for (v, st) in pr
        expr = st.expr
        if expr.head == :call && expr.args[1] == rand && expr.args[2] isa QuoteNode
            pr[v] = xcall(self, GlobalRef(parentmodule(@__MODULE__), :rand), expr.args[2 : end]...)
        end
    end
    IRTools.finish(pr)
end

# This pass inserts the return value of the call before any NoChange call nodes.
function substitute_get_value!(pr, v, st)
    expr = st.expr
    if expr.head == :call && expr.args[1] == rand && expr.args[2] isa QuoteNode
        pr[v] = xcall(GlobalRef(parentmodule(@__MODULE__), :record_cache!), self, expr.args[2])
    end
end

# This pass prunes the IR of any NoChange nodes.
function insert_cache_calls(ir)
    pr = IRTools.Pipe(ir)
    for (v, st) in pr
        if st.type == Change
            pr[v] = IRTools.Statement(st)
        else
            substitute_get_value!(pr, v, st)
        end
    end
    IRTools.finish(pr)
end

# This merges the meta from one piece of IR with the IR from a trace.
function reconstruct_ir(meta, tr)
    ir = IRTools.IR(copy(tr.defs), copy.(tr.blocks), copy(tr.lines), meta)
    ir
end

# Full pipeline.
pipeline(meta, tr) = tr |> tr -> (println(tr, "\n"); tr) |> insert_cache_calls |> strip_types |> rand_wrapper |> no_change_prune |> renumber |> tr -> reconstruct_ir(meta, tr) |> tr -> (println(tr); tr)
