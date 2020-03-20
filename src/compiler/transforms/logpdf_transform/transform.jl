@dynamo function logpdf_transform!(m...)
    ir = IR(m...)
    ir == nothing && return

    # Pipe allows incremental construction of IR.
    pr = Pipe(ir)
    var_map = Dict()
    insertions = []

    # Pass.
    for (v, st) in pr
        x = st.expr.args[1]

        # The dynamo sees a distribution.
        if x isa GlobalRef && x.name in dists
            vars = filter(x -> x isa Variable, st.expr.args)
            y = argument!(pr)
            var_map[v] = y
            z = insertafter!(pr, v, xcall(GlobalRef(Main.@__MODULE__, :logpdf), v, y))
            push!(insertions, z)
        end

        # The dynamo sees a rand call.
        if x isa GlobalRef && x.name == :rand
            vars = filter(x -> x isa Variable, st.expr.args)
            pr[v] = var_map[vars[1]]
        end

        # Else, the dynamo should recurse and create branches.
    end

    if !isempty(insertions)
        return!(pr, xcall(:+, insertions...))
    end

    # Finish.
    ir = finish(pr)
    deletearg!(ir, 1)
    ir = renumber(ir)
    return ir
end

logpdf_transform!(d::typeof(Normal), args...) = (println(d); d(args))
logpdf_transform!(lpdf::typeof(logpdf), args...) = lpdf(args)
