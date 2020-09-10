# Compute the Markov blanket of an address specified in rand calls by using a static reachability analysis.
function get_markov_blanket(reachability, addr)
    addr = unwrap(addr)
    v, check = get_variable_by_address(reachability, addr)
    !check && return nothing
    union(get_ancestors(reachability, v), get_successors(reachability, v))
end

function markov_blankets(reachability, addrs)
    d = Dict()
    for k in addrs
        d[k] = get_markov_blanket(reachability, k)
    end
    d
end

@inline check_change(st) = st.type == Change
@inline function check_removable(st)
    ex = st.expr
    ex isa Expr &&
    ex.head == :call && 
    (ex.args[1] isa IRTools.Inner.Self ||
    (ex.args[1] isa GlobalRef && ex.args[1].name == :record_cached!))
end

# This pass is a dataflow analysis which determines what deterministic statements can be safely removed. It leaves rand calls and record_cached! calls alone.
function no_change_prune(ir)
    us = IRTools.Inner.usecounts(ir)
    isused(x) = get(us, x, 0) > 0
    for v in reverse(keys(ir))
        if !isused(v) && !check_removable(ir[v])
            if isexpr(ir[v].expr)
                Mjolnir.effectful(Mjolnir.exprtype.((ir,), ir[v].expr.args)...) && continue
                map(v -> v isa Variable && (us[v] -= 1), ir[v].expr.args)
            elseif ir[v].expr isa Variable
                us[ir[v].expr] -= 1
            end
            delete!(ir, v)
        end
    end
    return ir
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
        pr[v] = xcall(GlobalRef(parentmodule(@__MODULE__), :record_cached!), self, expr.args[2])
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
@inline function pipeline(meta, tr, ks)
    tr = reconstruct_ir(meta, tr)
    display(tr)
    println()
    reachability = flow_analysis(tr)
    display(reachability)
    blankets = markov_blankets(reachability, ks)
    display(blankets)
    tr = insert_cache_calls(tr)
    tr = strip_types(tr)
    tr = rand_wrapper(tr)
    tr = no_change_prune(tr)
    tr = renumber(tr)
    tr
end
