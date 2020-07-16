# ------------ Utility selection, grads and params ------------ #

abstract type UtilitySelection <: Selection end

function fill_array!(val::T, arr::Vector{K}, f_ind::Int) where {K, T <: UtilitySelection}
    sorted_toplevel_keys = sort(collect(keys(val.utility)))
    sorted_tree_keys  = sort(collect(keys(val.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        v = val.utility[k]
        n = fill_array!(v, arr, idx)
        idx += n
    end
    for k in sorted_tree_keys
        n = fill_array!(get_sub(val, k), arr, idx)
        idx += n
    end
    idx - f_ind
end

function from_array(schema::T, arr::Vector{K}, f_ind::Int) where {K, T <: UtilitySelection}
    # Subtypes of UtilitySelection must support an empty constructor.
    sel = T()
    sorted_toplevel_keys = sort(collect(keys(schema.utility)))
    sorted_tree_keys  = sort(collect(keys(schema.tree)))
    idx = f_ind
    for k in sorted_toplevel_keys
        (n, v) = from_array(schema.utility[k], arr, idx)
        idx += n
        sel.utility[k] = v
    end
    for k in sorted_tree_keys
        (n, v) = from_array(get_sub(schema, k), arr, idx)
        idx += n
        sel.tree[k] = v
    end
    (idx - f_ind, sel)
end

# ------------ Gradients ------------ #

struct Gradients <: UtilitySelection
    tree::Dict{Address, Gradients}
    utility::Dict{Address, Any}
    Gradients() = new(Dict{Address, Gradients}(), Dict{Address, Any}())
    Gradients(tree, utility) = new(tree, utility)
end

has_grad(ps::Gradients, addr) = haskey(ps.utility, addr)

get_grad(ps::Gradients, addr) = getindex(ps.utility, addr)

has_sub(ps::Gradients, addr) = haskey(ps.tree, addr)

get_sub(ps::Gradients, addr) = getindex(ps.tree, addr)

function push!(ps::Gradients, addr, val)
    has_grad(ps, addr) && begin
        ps.utility[addr] += val
        return
    end
    ps.utility[addr] = val
end

Zygote.@adjoint Gradients(tree, utility) = Gradients(tree, utility), s_grad -> (nothing, nothing)

function merge(sel1::Gradients,
               sel2::Gradients)
    utility = merge(sel1.utility, sel2.utility)
    tree = Dict{Address, Gradients}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return Gradients(tree, utility)
end

+(a_grads::Gradients, b_grads::Gradients) = merge(a_grads, b_grads)

# ------------ Parameters, empty and learnable ------------ #

abstract type Parameters <: UtilitySelection end

struct NoParameters <: Parameters end

Parameters() = NoParameters()
has_param(np::NoParameters) = false
get_param(np::NoParameters) = error("(get_param) called on instance of NoParameters. No parameters!")
has_sub(np::NoParameters, addr) = false
get_sub(np::NoParameters, addr) = error("(get_sub) called on instance of NoParameters. No parameters!")

struct LearnableParameters <: Parameters
    tree::Dict{Address, LearnableParameters}
    utility::Dict{Address, Any}
    LearnableParameters() = new(Dict{Address, LearnableParameters}(), Dict{Address, Any}())
end

has_param(ps::LearnableParameters, addr) = haskey(ps.utility, addr)
get_param(ps::LearnableParameters, addr) = getindex(ps.utility, addr)
has_sub(ps::LearnableParameters, addr) = haskey(ps.tree, addr)
get_sub(ps::LearnableParameters, addr) = getindex(ps.tree, addr)

function push!(ps::LearnableParameters, addr, val)
    ps.utility[addr] = val
end

Zygote.@adjoint LearnableParameters(tree, utility) = LearnableParameters(tree, utility), s_grad -> (nothing, nothing)

function merge(sel1::LearnableParameters,
               sel2::LearnableParameters)
    utility = merge(sel1.utility, sel2.utility)
    tree = Dict{Address, Gradients}()
    for k in setdiff(keys(sel2.tree), keys(sel1.tree))
        tree[k] = sel2.tree[k]
    end
    for k in setdiff(keys(sel1.tree), keys(sel2.tree))
        tree[k] = sel1.tree[k]
    end
    for k in intersect(keys(sel1.tree), keys(sel2.tree))
        tree[k] = merge(sel1.tree[k], sel2.tree[k])
    end
    return LearnableParameters(tree, utility)
end

+(a::LearnableParameters, b::LearnableParameters) = merge(a, b)

# ------------ Trace to parameters ------------ #

function site_push!(chs::LearnableParameters, addr::Address, cs::LearnableSite)
    push!(chs, addr, cs.val)
end

function site_push!(chs::LearnableParameters, addr::Address, cs::GenericCallSite)
    subtrace = cs.trace
    subchs = LearnableParameters()
    for (k, v) in subtrace.calls
        site_push!(subchs, k, v)
    end
    for (k, v) in subtrace.params
        site_push!(subchs, k, v)
    end
    chs.tree[addr] = subchs
end

function push!(chs::LearnableParameters, tr::HierarchicalTrace)
    for (k, v) in tr.calls
        site_push!(chs, k, v)
    end
    for (k, v) in tr.params
        site_push!(chs, k, v)
    end
end

function get_parameters(tr::HierarchicalTrace)
    top = LearnableParameters()
    push!(top, tr)
    return top
end

function get_parameters(cl::GenericCallSite)
    top = LearnableParameters()
    push!(top, cl.trace)
    return top
end

# ------------ update_parameters links into Flux optimiser APIs ------------ #

function update_parameters(opt, a::LearnableParameters, b::Gradients)
    p_arr = array(a, Float64)
    gs_arr = array(b, Float64)
    update!(opt, p_arr, -gs_arr)
    return selection(a, p_arr)
end

# ------------ Pretty printing utility selections ------------ #

function collect!(par::T, addrs::Vector{Union{Symbol, Pair}}, chd::Dict{Union{Symbol, Pair}, Any}, chs::K) where {T <: Union{Symbol, Pair}, K <: UtilitySelection}
    for (k, v) in chs.utility
        push!(addrs, par => k)
        chd[par => k] = v
    end
    for (k, v) in chs.tree
        collect!(par => k, addrs, chd, v)
    end
end

function collect!(addrs::Vector{Union{Symbol, Pair}}, chd::Dict{Union{Symbol, Pair}, Any}, chs::K) where K <: UtilitySelection
    for (k, v) in chs.utility
        push!(addrs, k)
        chd[k] = v
    end
    for (k, v) in chs.tree
        collect!(k, addrs, chd, v)
    end
end

import Base.collect
function collect(chs::K) where K <: UtilitySelection
    addrs = Union{Symbol, Pair}[]
    chd = Dict{Union{Symbol, Pair}, Any}()
    collect!(addrs, chd, chs)
    return addrs, chd
end

function Base.display(chs::Gradients; show_values = false)
    println("  __________________________________\n")
    println("             Gradients\n")
    addrs, chd = collect(chs)
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

function Base.display(chs::LearnableParameters; show_values = false)
    println("  __________________________________\n")
    println("             Parameters\n")
    addrs, chd = collect(chs)
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
