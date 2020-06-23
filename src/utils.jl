# Log sum exp.
function lse(arr)
    max = maximum(arr)
    max == -Inf ? -Inf : max + log(sum(exp.(arr .- max)))
end

function lse(x1::Real, x2::Real)
    m = max(x1, x2)
    m == -Inf ? m : m + log(exp(x1 - m) + exp(x2 - m))
end

# Effective sample size.
function ess(lnw::Vector{Float64})
    log_ess = -lse(2. * lnw)
    return exp(log_ess)
end

# Normalize log weights.
function nw(lw::Vector{Float64})
    lt = lse(lw)
    lnw = lw .- lt
    return (lt, lnw)
end

# This is likely not a well-defined expectation. Investigate.
function average(trs::Vector{Trace}, addr::T) where T <: Address
    acc = 0.0
    trs = filter(trs) do tr
        addr in keys(tr.chm)
    end
    for tr in trs
        acc += tr.chm[addr].val
    end
    return acc/length(trs)
end

# Same goes for here.
function average(trs::Vector{Trace})
    d = Dict{Address, Tuple{Int, Real}}()
    for tr in trs
        for (k, coc) in collect(tr.chm)
            !(k in keys(d)) && begin
                d[k] = (1, coc.val)
                return
            end
            d[k] = (d[k][1] + 1, d[k][2] + coc.val)
        end
    end
    return Dict(map(collect(d)) do (k, v)
                    k => v[2]/v[1]
                end)
end

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
            collect!(k, addrs, chd, tr)
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

# Merge observations and a choice map.
function merge(tr::HierarchicalTrace,
    obs::ConstrainedHierarchicalSelection)
    tr_selection = chm(tr)
    merge!(tr_selection, obs)
    return tr_selection
end
