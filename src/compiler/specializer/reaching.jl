abstract type Analysis end
abstract type CallAnalysis <: Analysis end

struct ReachingAnalysis <: CallAnalysis
    reach::Dict
    ancestors::Dict
    sites::Set{Variable}
    addrs::Set{Any}
    map::Dict
    inv::Dict
    ir::IR
end

@inline function get_successors(ra::ReachingAnalysis, var)
    haskey(ra.reach, var) || return []
    ra.reach[var]
end

@inline function get_ancestors(ra::ReachingAnalysis, var)
    haskey(ra.ancestors, var) || return []
    ra.ancestors[var]
end

@inline function get_variable_by_address(ra::ReachingAnalysis, addr)
    haskey(ra.inv, addr) || return (nothing, false)
    return (getfield(ra, :inv)[addr], true)
end

function Base.display(ra::ReachingAnalysis)
    println("  __________________________________\n")
    println("              IR Reference\n")
    display(ra.ir)
    println("\n  __________________________________\n")
    println(" Addresses\n")
    println(ra.addrs)
    println()
    println(" Variable to address\n")
    display(ra.map)
    println()
    println(" Address to variable\n")
    display(ra.inv)
    println("\n Reachability\n")
    for x in ra.sites
        haskey(ra.reach, x) ? println(" $x => $(ra.reach[x])") : println(" $x")
    end
    println("\n Ancestors\n")
    for x in ra.sites
        haskey(ra.ancestors, x) ? println(" $x => $(ra.ancestors[x])") : println(" $x")
    end
    println("  __________________________________\n")
end

struct FallbackAnalysis <: CallAnalysis
end

mutable struct CallGraph <: Analysis
    addresses::Set{Address}
    dependencies::Dict{Address, Set{Address}}
    CallGraph(addrs, map) = new(addrs, map)
    CallGraph() = new(Dict{Address, Set{Address}}())
end

function Base.display(cg::CallGraph)
    println("  __________________________________\n")
    println("             Dependencies\n")
    for x in cg.addresses
        haskey(cg.dependencies, x) ? println(" $x => $(cg.dependencies[x])") : println(" $x")
    end
    println("  __________________________________\n")
end

# Reaching analysis.
function reaching!(reach::Vector{Variable}, p, var, ir)
    for (v, st) in ir
        st.expr isa Expr && begin
            if var in st.expr.args
                push!(reach, v)
                reaching!(reach, p, v, ir)
            end
        end
    end
end

function reaching(var::Variable, ir)
    reach = Variable[]
    for (v, st) in ir
        st.expr isa Expr && begin
            if var in st.expr.args
                push!(reach, v)
                reaching!(reach, var, v, ir)
            end
        end
    end
    return reach
end

function transitive_closure!(work, reach, s)
    for (k, v) in reach
        if s in v && k != s
            push!(work, k)
            transitive_closure!(work, reach, k)
        end
    end
end

@inline unwrap(sym::QuoteNode) = sym.value

function flow_analysis(ir)
    sites = Set(Variable[])
    addrs = Set(Any[])
    var_addr_map = Dict{Variable, Any}()
    reach = Dict{Variable, Any}()
    for (v, st) in ir
        MacroTools.postwalk(st) do e
            @capture(e, call_(sym_, args__))
            if call isa GlobalRef && call.name == :rand
                push!(sites, v)
                if sym isa QuoteNode
                    sym = sym.value
                end
                push!(addrs, sym)
                var_addr_map[v] = sym
                reach[v] = Set(reaching(v, ir))
            elseif call == rand
                push!(sites, v)
                push!(addrs, unwrap(sym))
                var_addr_map[v] = unwrap(sym)
                reach[v] = Set(reaching(v, ir))
            else
                push!(sites, v)
                reach[v] = Set(reaching(v, ir))
            end
            e
        end
    end
    ancestors = Dict()
    for s in sites
        work = Set(Variable[])
        for (k, v) in reach
            s in v && begin
                push!(work, k)
                transitive_closure!(work, reach, k)
            end
        end
        ancestors[s] = work
    end
    return ReachingAnalysis(reach, ancestors, sites, addrs, var_addr_map, Dict( v => k for (k, v) in var_addr_map), ir)
end

function dependency(a::Analysis)
    map = Dict{Any, Set{Any}}()
    for (k, vars) in a.reach
        depends = Any[]
        for v in vars
            haskey(a.map, v) && push!(depends, a.map[v].value)
        end
        map[a.map[k].value] = Set(depends)
    end
    addrs = [i.value for i in a.addrs]
    return CallGraph(Set(addrs), map)
end

# ------------ Driver ------------ #

# Returns the dependency analysis in call graph (tree) form.
function construct_graph!(parent, addr, call, type)
    ir = lower_to_ir(call, type)
    if control_flow_check(ir)
        graph = CallGraph()
    else
        analysis = flow_analysis(ir)
        dependencies = dependency(analysis)
        graph = CallGraph(dependencies)
    end
    parent[addr] = graph
end

function construct_graph(ir::IRTools.IR)
    if !control_flow_check(ir)
        graph = CallGraph()
    else
        analysis = flow_analysis(ir)
        dependencies = dependency(analysis)
        graph = dependencies
    end
    return graph
end

# Toplevel analysis driver. 
function construct_graph(call::Function, type...)
    ir = lower_to_ir(call, type...)
    construct_graph(ir)
end

function construct_graph(call::Function)
    ir = lower_to_ir(call)
    construct_graph(ir)
end
