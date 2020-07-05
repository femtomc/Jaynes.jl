# Pretty printing.
function Base.display(call::CallSite; 
                      fields::Array{Symbol, 1} = [:val],
                      show_full = false)
    println("  __________________________________\n")
    println("               Playback\n")
    map(fieldnames(CallSite)) do f
        val = getfield(call, f)
        typeof(val) <: Dict{Address, ChoiceSite} && begin 
            vals = collect(val)
            if length(vals) > 5 && !show_full
                map(vals[1:5]) do (k, v)
                    println(" $(k)")
                    map(fieldnames(ChoiceSite)) do nm
                        !(nm in fields) && return
                        println("          $(nm)  = $(getfield(v, nm))")
                    end
                    println("")
                end
                println("                  ...\n")
                println("  __________________________________\n")
                return
            else
                map(vals) do (k, v)
                    println(" $(k)")
                    map(fieldnames(ChoiceSite)) do nm
                        !(nm in fields) && return
                        println("          $(nm)  = $(getfield(v, nm))")
                    end
                    println("")
                end
                println("  __________________________________\n")
                return
            end
        end
        typeof(val) <: Real && begin
            println(" $(f) : $(val)\n")
            return
        end
        println(" $(f) : $(typeof(val))\n")
    end
    println("  __________________________________\n")
end

function collect!(par::T, addrs::Vector{Union{Symbol, Pair}}, chd::Dict{Union{Symbol, Pair}, Any}, tr::Trace) where T <: Union{Symbol, Pair}
    for (k, v) in tr.chm
        if v isa ChoiceSite
            push!(addrs, par => k)
            chd[par => k] = v.val
        elseif v isa CallSite
            collect!(par => k, addrs, chd, v.trace)
        elseif v isa VectorizedCallSite
            for i in 1:length(v.subtraces)
                collect!(par => k => i, addrs, chd, v.subtraces[i])
            end
        end
    end
    return addrs
end

function collect!(addrs::Vector{Union{Symbol, Pair}}, chd::Dict{Union{Symbol, Pair}, Any}, tr::Trace)
    for (k, v) in tr.chm
        if v isa ChoiceSite
            push!(addrs, k)
            chd[k] = v.val
        elseif v isa CallSite
            collect!(k, addrs, chd, v.trace)
        elseif v isa VectorizedCallSite
            for i in 1:length(v.subtraces)
                collect!(k => i, addrs, chd, v.subtraces[i])
            end
        end
    end
end

import Base.collect
function collect(tr::Trace)
    addrs = Union{Symbol, Pair}[]
    chd = Dict{Union{Symbol, Pair}, Any}()
    collect!(addrs, chd, tr)
    return addrs, chd
end

function Base.display(tr::Trace; show_values = false)
    println("  __________________________________\n")
    println("               Addresses\n")
    addrs, chd = collect(tr)
    if show_values
        for a in addrs
            println(" $(a) : $(chd[a])")
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println("  __________________________________\n")
end
