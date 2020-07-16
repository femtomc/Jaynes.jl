# Pretty printing.
function Base.display(call::C; 
                      fields::Array{Symbol, 1} = [:val],
                      show_full = false) where C <: CallSite
    println("  __________________________________\n")
    println("               Playback\n")
    println(" type : $C\n")
    map(fieldnames(C)) do f
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

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::HierarchicalTrace, meta) where T <: Union{Symbol, Int, Pair}
    for (k, v) in tr.choices
        push!(addrs, par => k)
        chd[par => k] = v.val
    end
    for (k, v) in tr.calls
        if v isa GenericCallSite
            collect!(par => k, addrs, chd, v.trace, meta)
        elseif v isa VectorizedSite
            for i in 1:length(v.trace.subrecords)
                collect!(par => k => i, addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
    for (k, v) in tr.params
        push!(meta, par => k)
        push!(addrs, par => k)
        chd[par => k] = v.val
    end
end

function collect!(par::T, addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorizedTrace, meta) where T <: Union{Symbol, Int, Pair}
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, par => k)
            chd[par => k] = v.val
        elseif v isa GenericCallSite
            collect!(par => k, addrs, chd, v.trace, meta)
        elseif v isa VectorizedSite
            for i in 1:length(v.trace.subrecords)
                collect!(par => k => i, addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
    for (k, v) in tr.params
        push!(meta, k)
        push!(addrs, k)
        chd[k] = v.val
    end
end

function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::HierarchicalTrace, meta)
    for (k, v) in tr.choices
        push!(addrs, k)
        chd[k] = v.val
    end
    for (k, v) in tr.calls
        if v isa GenericCallSite
            collect!(k, addrs, chd, v.trace, meta)
        elseif v isa VectorizedSite
            for i in 1:length(v.trace.subrecords)
                collect!(k => i, addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
    for (k, v) in tr.params
        push!(meta, k)
        push!(addrs, k)
        chd[k] = v.val
    end
end

function collect!(addrs::Vector{Any}, chd::Dict{Any, Any}, tr::VectorizedTrace, meta)
    for (k, v) in enumerate(tr.subrecords)
        if v isa ChoiceSite
            push!(addrs, k)
            chd[k] = v.val
        elseif v isa GenericCallSite
            collect!(k, addrs, chd, v.trace, meta)
        elseif v isa VectorizedSite
            for i in 1:length(v.trace.subrecords)
                collect!(k => i, addrs, chd, v.trace.subrecords[i].trace, meta)
            end
        end
    end
    for (k, v) in tr.params
        push!(meta, k)
        push!(addrs, k)
        chd[k] = v.val
    end
end

import Base.collect
function collect(tr::Trace)
    addrs = Any[]
    chd = Dict{Any, Any}()
    meta = Any[]
    collect!(addrs, chd, tr, meta)
    return addrs, chd, meta
end

function Base.display(tr::Trace; 
                      show_values = false, 
                      show_types = false)
    println("  __________________________________\n")
    println("               Addresses\n")
    addrs, chd, meta = collect(tr)
    if show_values
        for a in addrs
            if a in meta
                println(" (Learnable)  $(a) : $(chd[a])")
            else
                println(" $(a) : $(chd[a])")
            end
        end
    elseif show_types
        for a in addrs
            if a in meta
                println(" (Learnable)  $(a) : $(typeof(chd[a]))")
            else
                println(" $(a) : $(typeof(chd[a]))")
            end
        end
    elseif show_types && show_values
        for a in addrs
            if a in meta
                println(" (Learnable)  $(a) : $(chd[a]) : $(typeof(chd[a]))")
            else
                println(" $(a) : $(chd[a]) : $(typeof(chd[a]))")
            end
        end
    else
        for a in addrs
            if a in meta
                println(" (Learnable)  $(a)")
            else
                println(" $(a)")
            end
        end
    end
    println("  __________________________________\n")
end
