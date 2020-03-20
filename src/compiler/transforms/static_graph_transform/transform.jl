# Dependency graph.
const DependencyGraph = MetaDiGraph
add_v!(n::Any, g::DependencyGraph) = !(n in [get_prop(g, i, :name) for i in vertices(g)]) && add_vertex!(g, :name, n)
loc_tuple = (depth, x) -> (l = getlocation(x.info); (l.block, l.line))

# Utilities for moving around the IR.
head(p::Pair) = p[1]
tail(p::Pair) = p[2]

function remainder(var::Variable, ir)::Array{Pair{Variable, Statement}}
    map(v -> v => ir[v], filter(x -> x.id > var.id, keys(ir)))
end

function var_check(var::Variable, st::Statement)
    var in st.expr.args && return true
end

function dependents(var::Variable, ir)::Array{Pair{Variable, Statement}}
    rmdr = remainder(var, ir)
    a = Array{Pair{Variable, Statement}}([])
    length(rmdr) == 0 ? nothing : map(x -> var_check(var, tail(x)) ? push!(a, x) : nothing, rmdr)
    return a
end

function dependency_graph(ir)::DependencyGraph
    g = MetaDiGraph()
    set_indexing_prop!(g, :name)
    map(x -> add_v!(x, g), keys(ir))
    for var in keys(ir)
        map(x -> add_edge!(g, g[var, :name], g[head(x), :name]), dependents(var, ir))
    end
    return g
end
