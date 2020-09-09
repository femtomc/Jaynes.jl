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

# This pass inserts the return value of the call before any NoChange call nodes.
function substitute_return!(pr, v, st)
    expr = st.expr
    if expr.head == :call && expr.args[1] == rand
        
        # TODO: fix.
        pr[v] = xcall(IRTools.self, GlobalRef(parentmodule(@__MODULE__), :rand), expr.args[2 : end]...)

    else
        call = expr.args[1]
        
        # TODO: fix.
        call = string(call)[findlast(x -> x == '.', string(call)) + 1 : end]

        new = Expr(expr.head, GlobalRef(parentmodule(@__MODULE__), Symbol(call)), expr.args[2 : end]...)
        pr[v] = IRTools.Statement(new, type = Any)
    end
end

# This pass prunes the IR of any NoChange nodes.
function prune(ir)
    pr = IRTools.Pipe(ir)
    for (v, st) in pr
        if st.type == Change
            pr[v] = IRTools.Statement(st)
        else
            substitute_return!(pr, v, st)
        end
    end
    IRTools.finish(pr)
end
