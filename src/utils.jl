function lse(arr)
    max = maximum(arr)
    max == -Inf ? -Inf : max + log(sum(exp.(arr .- max)))
end

function lse(x1::Real, x2::Real)
    m = max(x1, x2)
    m == -Inf ? m : m + log(exp(x1 - m) + exp(x2 - m))
end

function average(trs::Vector{Trace}, addr::T) where T <: Union{Symbol, Pair}
    acc = 0.0
    trs = filter(trs) do tr
        addr in keys(tr.chm)
    end
    for tr in trs
        acc += tr.chm[addr].val
    end
    return acc/length(trs)
end

function average(trs::Vector{Trace})
    d = Dict{Union{Symbol, Pair}, Tuple{Int, Real}}()
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
function Base.println(tr::Trace, fields::Array{Symbol, 1})
    println("/---------------------------------------")
    map(fieldnames(Trace)) do f
        f == :stack && return
        val = getfield(tr, f)
        typeof(val) <: Dict{Union{Symbol, Pair}, ChoiceOrCall} && begin 
            map(collect(val)) do (k, v)
                println("| <> $(k) <>")
                map(fieldnames(ChoiceOrCall)) do nm
                    !(nm in fields) && return
                    println("|          $(nm)  = $(getfield(v, nm))")
                end
                println("|")
            end
            return
        end
        typeof(val) <: Dict{Union{Symbol, Pair}, Real} && begin 
            println("| $(f) __________________________________")
            map(collect(val)) do (k, v)
                println("|      $(k) : $(v)")
            end
            println("|")
            return
        end
        println("| $(f) : $(val)\n|")
    end
    println("\\---------------------------------------")
end

# Merge observations and a choice map.
function merge(obs::Dict{Address, K},
               chm::Dict{Address, ChoiceOrCall}) where K
    obs_ks = collect(keys(obs))
    chm_ks = collect(keys(chm))
    out = Dict{Address, K}()
    for k in chm_ks
        k in obs_ks && error("SupportError: proposal has address on observed value.")
        out[k] = chm[k].val
    end
    for k in obs_ks
        out[k] = obs[k]
    end
    return out
end

# Make obs.
function constraints(obs::Array{Tuple{T, K}, 1}) where {T <: Address, K}
    d = Dict{Address, K}()
    for (k, v) in obs 
        d[k] = v
    end
    return d
end
