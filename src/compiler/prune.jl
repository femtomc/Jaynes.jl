# This pass inserts the return value of the call before any NoChange call nodes.
function substitute_return!(st)
    println(st.type)
    st
end

# This pass prunes the IR of any NoChange nodes.
function prune!(ir)
    pr = IRTools.Pipe(ir)
    for (v, st) in pr
        st.type == UndefinedChange && continue
        pr[v] = substitute_return!(st)
    end
    IRTools.finish(pr)
end
