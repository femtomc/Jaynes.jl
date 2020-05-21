function (tr::Trace)(call::typeof(rand), 
                     addr::T, 
                     dist::Distribution) where T <: Union{Symbol, Pair}
    sample = rand(dist)
    addr = tr.stack[end] => addr
    addr in keys(tr.obs) && begin
        sample = tr.obs[addr]
    end
    lpdf = logpdf(dist, sample)
    addr in keys(tr.chm) && error("AddressError: each address within a call must be unique. Found duplicate $(addr).")
    tr.chm[addr] = ChoiceOrCall(sample, lpdf, dist)
    tr.score += lpdf
    return sample
end

# TODO: fix.
function prepass(ir::IR)
    truth = true
    for (v, st) in ir
        expr = st.expr
        expr isa Expr && 
        expr.head == :call && 
        expr.args[1] isa GlobalRef &&
        expr.args[1].name == :rand && begin
            expr.args[2] isa QuoteNode && return truth
            truth = false
        end
    end
    return truth
end

@dynamo function (tr::Trace)(a...)
    ir = IR(a...)
    ir == nothing && return
    #check = prepass(ir)
    #!check && error("AddressError: calls to rand must be annotated with a unique address.")
    recurse!(ir)
    for (v, st) in ir
        expr = st.expr
        expr isa Expr && expr.head == :call && expr.args[2] isa GlobalRef && begin
            insert!(ir, v, xcall(push!, self, QuoteNode(expr.args[2].name)))
            insertafter!(ir, v, xcall(pop!, self))
        end
    end
    return ir
end
