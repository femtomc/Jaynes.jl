# Graph IR - a representation which can be packaged and sent into the runtime execution context for specialization.

abstract type GraphIR end

# Static analysis allows construction of the highly-efficient GraphTrace for method bodies which support it. 
# This is constructed by determining where randomness flows using a reaching analysis. 
# When used in iterative contexts, the dependence information can be used to identify argdiff-style information automatically in any call. 
# If the analysis fails for a call - the fallback is the hierarchical (or vector) trace.

abstract type Analysis end
abstract type CallAnalysis <: Analysis end

struct StaticAnalysis <: CallAnalysis
    reach::Dict
    sites::Vector{Variable}
    addrs::Vector{QuoteNode}
    map::Dict
end

struct FallbackAnalysis <: CallAnalysis
end

mutable struct CallGraph{T} <: Analysis
    expanded::Dict{Address, CallGraph}
    dependencies::T
    CallGraph(map::T) where T = new{T}(Dict{Address, CallGraph}(), map)
    CallGraph() = new{Nothing}(Dict{Address, CallGraph}(), nothing)
end

# ---- Analysis ---- #

function lower_to_ir(call, type...)
    sig = Tuple{typeof(call), type...}
    m = meta(sig)
    ir = IR(m)
    return ir
end

function control_flow_check(ir)
    length(ir.blocks) > 1 && return false
    return true
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

function flow_analysis(ir)
    sites = Variable[]
    addrs = QuoteNode[]
    var_sym_map = Dict{Variable, QuoteNode}()
    reach = Dict{Variable, Any}()
    for (v, st) in ir
        MacroTools.postwalk(st) do e
            @capture(e, call_(sym_, args__))
            call isa GlobalRef && call.name == :rand && begin
                push!(sites, v)
                push!(addrs, sym)
                var_sym_map[v] = sym
                r = reaching(v, ir)
                !isempty(r) && begin
                    reach[v] = r
                end
            end
            e
        end
    end
    return StaticAnalysis(reach, sites, addrs, var_sym_map)
end

function dependency(a::Analysis)
    map = Dict{Symbol, Set{Symbol}}()
    for (k, vars) in a.reach
        depends = Symbol[]
        for v in vars
            haskey(a.map, v) && push!(depends, a.map[v].value)
        end
        map[a.map[k].value] = Set(depends)
    end
    return CallGraph(map)
end

# ---- Driver ---- #

# Recurses into `rand` calls with function calls (in all branches!). 
# If the call has control flow which is not unrollable at compile-time, falls back on no analysis.
# Returns the dependency analysis in call graph (tree) form.
function construct_graph!(parent, addr, call, type)
    ir = lower_to_ir(call, type)
    if control_flow_check(ir)
        @info "In IR at address $(addr): static analysis error.\nFalling back on hierarchical representation."
        graph = CallGraph()
    else
        analysis = flow_analysis(ir)
        dependencies = dependency(analysis)
        graph = CallGraph(dependencies)
    end
    # Recurse
    
    # Mutate.
    parent[addr] = graph
end

# Toplevel analysis driver. 
function construct_graph(call, type)
    ir = lower_to_ir(call, type)
    if !control_flow_check(ir)
        @info "In $(stacktrace()[2]): analysis error.\nFalling back on hierarchical representation."
        graph = CallGraph()
    else
        analysis = flow_analysis(ir)
        dependencies = dependency(analysis)
        graph = CallGraph(dependencies)
    end

    # Now recurse into rand calls.

    return graph
end
