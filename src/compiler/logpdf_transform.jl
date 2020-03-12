@dynamo function logpdf_transform!(m...)
    ir = IR(m...)
    pr = Pipe(ir)
    arg_map = Dict()
    var_map = Dict()
    insertions = []

    # Pass.
    for (v, st) in pr
        x = st.expr.args[1]
        if x isa GlobalRef && x.name in dists
            vars = filter(x -> x isa Variable, st.expr.args)
            y = argument!(pr)
            var_map[v] = y
            z = insertafter!(pr, v, xcall(GlobalRef(Main.@__MODULE__, :logpdf), v, y))
            push!(insertions, z)
        end

        if x isa GlobalRef && x.name == :rand
            vars = filter(x -> x isa Variable, st.expr.args)
            pr[v] = var_map[vars[1]]
        end
    end
    return!(pr, xcall(:+, insertions...))

    # Finish.
    ir = finish(pr)
    deletearg!(ir, 1)
    return renumber(ir)
end
