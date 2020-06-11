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

function collect_addrs!(par::T, addrs::Vector{Union{Symbol, Pair}}, tr::Trace) where T <: Union{Symbol, Pair}
    for (k, v) in tr.chm
        push!(addrs, par => k)
        if v isa CallSite
            collect_addrs!(par => k, addrs, v.trace)
        end
    end
    return addrs
end

function collect_addrs!(addrs::Vector{Union{Symbol, Pair}}, tr::Trace)
    for (k, v) in tr.chm
        push!(addrs, k)
        if v isa CallSite
            collect_addrs!(k, addrs, tr)
        end
    end
end

function collect_addrs(tr::Trace)
    addrs = Union{Symbol, Pair}[]
    collect_addrs!(addrs, tr)
    return addrs
end

function Base.display(tr::Trace)
    println("  __________________________________\n")
    println("               Addresses\n")
    addrs = collect_addrs(tr)
    map(addrs) do a
        println(" ", a)
    end
    println("  __________________________________\n")
end

# Merge observations and a choice map.
function merge(obs::Dict{Address, K},
    chm::Dict{Address, ChoiceSite}) where K
    cons = copy(obs)
    for (k, v) in chm
        haskey(cons, k) && error("SupportError: proposal has address on observed value.")
        cons[k] = v.val
    end
    return cons
end

# Take observations and return a selection.
function selection(obs::Array{Tuple{T, K}, 1}) where {T <: Address, K}
    d = Dict{Address, K}()
    for (k, v) in obs 
        d[k] = v
    end
    return ConstrainedSelection(d)
end

function selection(obs::Array{T, 1}) where {T <: Address}
    return UnconstrainedSelection([first(el) for el in obs])
end

function selection()
    return EmptySelection()
end
