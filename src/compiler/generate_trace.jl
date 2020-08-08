@generated function compile_function(fn, args...)
    T = Tuple{fn, args...}
    m = IRTools.meta(T)
    m isa Nothing && return
    ir = IRTools.IR(m)
    g = construct_graph(ir)
    fieldnames = Expr[]
    for a in g.addresses
        push!(fieldnames, Expr(:(::), a, :RecordSite))
    end
    tr_name = gensym("static")
    expr = Expr(:struct, false, :Foo, Expr(:block, fieldnames...))
    expr = MacroTools.postwalk(rmlines ∘ unblock, expr)
    expr
end
