module JaynesCore

using IRTools
using IRTools: IR, @dynamo, recurse!, xcall, self, insertafter!, insert!
using Distributions

# -------------------- CORE ---------------------- #

import Base.rand
rand(addr::Symbol, d::Type, args::Tuple) = rand(d(args...))
rand(addr::Symbol, fn::Function, args::Tuple) = fn(args...)

struct ChoiceOrCall{T}
    val::T
    lpdf::Float64
    d::Distribution
end

const Address = Union{Symbol, Pair}
mutable struct Trace
    chm::Dict{Address, ChoiceOrCall}
    obs::Dict{Address, Number}
    stack::Vector{Any}
    score::Float64
    Trace() = new(Dict{Symbol, Any}(), Dict{Symbol, Any}(), Core.TypeName[], 0.0)
end

function Trace(chm::Dict{T, K}) where {T <: Union{Symbol, Pair}, K <: Number}
    tr = Trace()
    tr.obs = obs
    return tr
end

import Base: push!, pop!
function push!(tr::Trace, call::Symbol)
    call == :rand && begin
        push!(tr.stack, tr.stack[end])
        return
    end
    isempty(tr.stack) && begin
        push!(tr.stack, call)
        return
    end
    push!(tr.stack, tr.stack[end] => call)
end

function pop!(tr::Trace)
    pop!(tr.stack)
end

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
        typeof(val) <: Dict{Union{Symbol, Pair}, Number} && begin 
            println("| $(f) __________________________________")
            map(collect(val)) do (k, v)
                println("|      $(k) : $(v)")
            end
            println("|")
            return
        end
        println("| $(f) : $(val)\n|")
    end
    println("\\----------------------------------------")
end

macro trace(call)
    quote
        tr = Trace()
        tr() do
            $call
        end
        tr
    end
end

macro trace(obs, call)
    quote
        tr = Trace($obs)
        tr() do
            $call
        end
        tr
    end
end

# ------------------------- IR ------------------------- #

function (tr::Trace)(call::typeof(rand), addr::T, d::Type, args::Tuple) where T <: Union{Symbol, Pair}
    dist = d(args...)
    sample = rand(dist)
    addr = tr.stack[end] => addr
    addr in keys(tr.obs) && begin
        sample = tr.obs[addr]
    end
    lpdf = logpdf(dist, sample)
    addr in keys(tr.chm) && error("AddressError: each address within a call must be unique. Found duplicate $(addr).")
    tr.chm[addr] = ChoiceOrCall(sample, lpdf, dist)
    tr.score += lpdf
    return sample
end

function prepass(ir::IR)
    truth = true
    for (v, st) in ir
        expr = st.expr
        expr isa Expr && 
        expr.head == :call && 
        expr.args[1] isa GlobalRef &&
        expr.args[1].name == :rand && begin
            expr.args[2] isa QuoteNode && return truth
            truth = false
        end
    end
    return truth
end

@dynamo function (tr::Trace)(a...)
    ir = IR(a...)
    ir == nothing && return
    #check = prepass(ir)
    #!check && error("AddressError: calls to rand must be annotated with a unique address.")
    recurse!(ir)
    for (v, st) in ir
        expr = st.expr
        expr isa Expr && expr.head == :call && expr.args[2] isa GlobalRef && begin
            insert!(ir, v, xcall(push!, self, QuoteNode(expr.args[2].name)))
            insertafter!(ir, v, xcall(pop!, self))
        end
    end
    return ir
end


# -------------------- FUN ---------------------- #

function bar(x::Float64)
    for i in 1:10
        q = rand(:q => i, Normal, (x, 1.0))
    end
    z = rand(:z, Normal, (x, 1.0))
    return z
end

function foo()
    y = rand(:y, Normal, (0.0, 1.0))
    x = bar(y)
    return x
end

tr = @trace foo()
println(tr, [:val])

# Constraints.
obs = Dict((:foo => :y) => 7.0)
tr = @trace obs foo()
println(tr, [:val])

end # module
