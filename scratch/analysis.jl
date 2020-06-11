module GraphIRScratch

using IRTools
using IRTools: blocks, typed_meta, IR, Variable
using MacroTools
using Distributions
Address = Union{Symbol, Pair}

function bar(z::Float64)
    z = rand(:z, Normal, (z, 5.0))
    return z
end

function foo(x::Int)
    z = rand(:z, Normal, (0.0, 1.0))
    y = z
    q = rand(:q, Normal, (y, 3.0))
    l = rand(:l, Normal, (q, z))
    r = rand(:r, Normal, (l, y))
    m = rand(:bar, bar, (r, ))
    return rand(:bar, bar, (q, ))
end

# ---- Analysis ----

struct Analysis
    reach::Dict
    sites::Vector{Variable}
    addrs::Vector{QuoteNode}
    map::Dict
end

function typed_ir(call, type)
    sig = Tuple{typeof(call), type}
    m = typed_meta(sig)
    ir = IR(m)
    return ir
end

function control_flow_check(ir)
    length(ir.blocks) > 1 && return false
    return true
end

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
    return Analysis(reach, sites, addrs, var_sym_map)
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
    return map
end

function toplevel(call, type)
    ir = typed_ir(call, type)
    !control_flow_check(ir) && begin
        @info "In $(stacktrace()[2]): analysis error.\nFalling back on hierarchical representation."
        return (false, nothing)
    end
    analysis = flow_analysis(ir)
    dependencies = dependency(analysis)
    println(dependencies)
    return (true, ir)
end


b, ir = toplevel(foo, Int)
println(ir)

end # module
