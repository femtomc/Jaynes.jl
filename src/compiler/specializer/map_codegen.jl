function generate_map(call, cg)
    fields = []
    for addr in cg.addresses
        var = gensym()
        push!(fields, (Expr(:(::), addr, var), var))
    end
    label = Symbol(gensym(call), :StaticMap)
    struct_expr = Expr(:block,
                Expr(:struct, 
                     false, 
                     Expr(:curly, label, [x[2] for x in fields]...), 
                     Expr(:block, [x[1] for x in fields]...)))

    method_expr = quote
        function shallow_iterator(m::$label)
            map(fieldnames($label)) do f
                (f, getfield(m, f))
            end
        end

        function getindex(x::$label, addr)
            getfield(x, addr)
        end
    end

    declaration = quote
        $struct_expr
        $method_expr

        println("Specialized method call.")
    end
    MacroTools.postwalk(rmlines ∘ unblock, declaration)
end

function _specialize(expr)
    MacroTools.@capture(expr, call_(args__)) || error("Transform only applies to calls.")
    cg = construct_graph(eval(call), args...)
    expr = generate_map(call, cg)
end

macro specialize(expr)
    trans = _specialize(expr)
    println(trans)
    trans
end
