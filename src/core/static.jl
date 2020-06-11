# A static analysis allows construction of the highly-efficient GraphTrace. This is constructed by determining where randomness flows. When used in iterative contexts, the dependence information can be used to identify argdiff-style information automatically in any call. If the analysis fails for a call - the fallback is the hierarchical (or vector) trace.

struct Analysis
    reach::Dict
    sites::Vector{Variable}
    addrs::Vector{QuoteNode}
    map::Dict
end

function lower_to_typed_ir(call, type)
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
    return (true, dependencies)
end

Cassette.@context GraphAnalysisCtx

@inline function Cassette.overdub(ctx::GraphAnalysisCtx,
                                  c::typeof(rand),
                                  addr::T,
                                  call::Function,
                                  args...) where T <: Address

    test, dependencies = toplevel(call, typeof(args...))
    ret = recurse(rec_ctx, call, args...)
    ctx.metadata.tr.chm[addr] = CallSite(rec_ctx.metadata.tr, 
                                         call, 
                                         args..., 
                                         ret)
    return ret
end

