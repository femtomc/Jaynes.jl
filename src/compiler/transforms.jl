# Get the IR variable corresponding to an address.
function get_variable_by_address(ir, addr)
    for (v, st) in ir
        check = false
        MacroTools.postwalk(st.expr) do ex
            if ex isa QuoteNode && addr == ex.value
                check = true
            end
            ex
        end
        check && return (v, true)
    end
    (nothing, false)
end

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

function convert_to_blanket(dep, v)
    ch = dep[v]
    pars = filter(keys(dep)) do k
        v in dep[k]
    end
    union(ch, pars)
end

# Compute the Markov blanket of an address specified in rand calls.
function get_markov_blanket(ir, addr)
    v, check = get_variable_by_address(ir, addr)
    !check && return nothing
    dep = dependencies(ir)
    convert_to_blanket(dep, v)
end

function markov_blankets(ir, addrs)
    d = Dict()
    for k in addrs
        d[k] = get_markov_blanket(ir, k)
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
    blankets = markov_blankets(tr, ks)
    flows = flow_analysis(tr)
    println()
    display(flows)
    println()
    display(blankets)
    println()
    display(tr)
    println()
    tr = insert_cache_calls(tr)
    tr = strip_types(tr)
    tr = rand_wrapper(tr)
    tr = no_change_prune(tr)
    tr = renumber(tr)
    display(tr)
    tr
end
