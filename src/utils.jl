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

# Regenerate and update.
function visited_walk!(discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, par_addr => addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, tr::T, vs::VisitedSelection) where T <: Trace
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(par_addr, discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
        
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(par_addr => addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
end

# Toplevel.
function visited_walk!(tr::T, vs::VisitedSelection) where T <: Trace
    discard = ConstrainedHierarchicalSelection()
    d_score = 0.0
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
       
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
    tr.score -= d_score
    return discard
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
            collect!(k, addrs, chd, v.trace)
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
    tr_selection = selection(tr)
    merge!(tr_selection, obs)
    return tr_selection
end

