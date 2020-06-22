module GraphIRScratch

using IRTools
using IRTools: blocks
using Mjolnir
using Mjolnir: Basic, AType, Const, abstract
using Distributions
Address = Union{Symbol, Pair}

import Base.rand
rand(addr::T, d::Type, args) where T <: Address = rand(d(args...))
rand(addr::T, d::Function, args) where T <: Address = d(args...)
rand(addr::T, lit::K) where {T <: Address, K <: Union{Number, AbstractArray}} = lit

Mjolnir.@abstract Basic rand(addr::Symbol, d::Type, args) = Any
Mjolnir.@abstract Basic rand(addr::Address, d::Type, args) = Any

# Static pass which checks for mutation, iteration, or any branching.
function pure_pass(ir)
    for (v, st) in ir
        ex = st.expr
        ex isa Expr && ex.head == :call && ex.args[1] isa GlobalRef && begin
            callname = ex.args[1].name
            callname == :push! && error("GraphDSL: mutation is not supported.")
            callname == :setfield! && error("GraphDSL: mutation is not supported.")
            callname == :setindex! && error("GraphDSL: mutation is not supported.")
        end
    end
    !(length(blocks(ir)) == 1) && error("GraphDSL: control flow is not supported.")
end

function compile(fn::Function, args::Tuple)
    ir = Mjolnir.@trace fn(args...)
    println(ir)
end

# --------- TEST --------- #

function bar(z::Float64)
    z = rand(:z, Normal, (z, 5.0))
    return z
end

function foo(x::Int)
    z = rand(:z, Normal, (0.0, 1.0))
    y = 0.0
    for i in 1:3
        y = rand(:y => i, Normal, (y, 1.0))
    end
    q = rand(:q, Normal, (y, 3.0))
    return rand(:bar, bar, (q, ))
end

compile(foo, (7, ))

end # module
