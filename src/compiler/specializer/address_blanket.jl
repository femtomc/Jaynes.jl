function direct_deps!(reach::Vector{Variable}, p, var, ir)
    for (v, st) in ir
        st.expr isa Expr && begin
            if var in st.expr.args
                push!(reach, v)
                if st.expr.head == :call &&
                    st.expr.args[1] isa GlobalRef && st.expr.args[1].name == :trace
                    # blocked
                else
                    direct_deps!(reach, p, v, ir)
                end
            end
        end
    end
end
function direct_deps(var::Variable, ir)
    reach = Variable[]
    for (v, st) in ir
        st.expr isa Expr && begin
            if var in st.expr.args
                push!(reach, v)
                direct_deps!(reach, var, v, ir)
            end
        end
    end
    return reach
end
function direct_dep_analysis(ir)
    sites = Variable[]
    addrs = QuoteNode[]
    var_sym_map = Dict{Variable, QuoteNode}()
    reach = Dict{Variable, Any}()
    for (v, st) in ir
        Jaynes.MacroTools.postwalk(st) do e
            Jaynes.@capture(e, call_(sym_, args__))
            call isa GlobalRef && call.name == :trace && begin
                push!(sites, v)
                push!(addrs, sym)
                var_sym_map[v] = sym
                r = direct_deps(v, ir)
                !isempty(r) && begin
                    reach[v] = r
                end
            end
            e
        end
    end
    return StaticAnalysis(reach, sites, addrs, var_sym_map)
end

function markov_blanket(ir::IRTools.IR)
    ds = dependency(direct_dep_analysis(ir))

    addrs = ds.addresses
    G = copy(ds.dependencies)

    # add empty diagonal
    for p in addrs
        if !haskey(G, p)
            G[p] = Set{Address}()
        end
    end

    # moralize
    parents = [[i for (i, js) in pairs(G) if k in js] for k in addrs]
    for pars in parents
        for p in pars
            for q in pars
                p == q && continue
                push!(G[p], q)
            end
        end
    end

    # symmetrize and add diagonal
    for (p, qs) in G
        for q in qs
            p == q && continue
            push!(G[q], p)
        end
        push!(G[p], p)
    end
    G
end

function markov_blanket(call::Function, type...)
    ir = lower_to_ir(call, type...)
    markov_blanket(ir)
end
