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
    map(trs) do tr
        acc += tr.chm[addr].val
    end
    return acc/length(trs)
end

function average(trs::Vector{Trace})
    d = Dict{Union{Symbol, Pair}, Tuple{Int, Real}}()
    map(trs) do tr
        map(collect(tr.chm)) do (k, coc)
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

# Make obs.
function constraints(obs::Array{Tuple{T, K}}) where {T <: Union{Symbol, Pair}, K <: Real}
    return Dict{Address, Real}(map(obs) do (k, v)
                            k => v
                        end)
end
