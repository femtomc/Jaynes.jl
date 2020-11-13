resolve(x) = x
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)

function prepare_ir!(ir)
    for (v, st) in ir
        isexpr(st.expr) || continue 
        ir[v] = stmt(Expr(st.expr.head, map(resolve, st.expr.args)...); type = Union{})
    end
    ir
end

function infercall!(env, v, st, ir)
    args = map(st.expr.args) do a
        k = unwrap(a)
        get(env, k, k)
    end
    t = abst(args...)
    env[v] = t
    ir[v] = stmt(st.expr; type = t)
end

function infer!(ir)
    env = Dict()
    for (v, st) in ir
        isexpr(st.expr, :call) && infercall!(env, v, st, ir)
    end
    ir
end
