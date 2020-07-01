module Jaynes2

module Mixtape

using IRTools
using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!, isexpr, self, argument!, var

const Address = Union{Symbol, Pair{Symbol, Int}}

import Base.rand
rand(addr::Address, d::Distribution{T}) where T = rand(d)

abstract type MixTable end
@dynamo function (mx::MixTable)(a...)
  ir = IR(a...)
  ir == nothing && return
  recurse!(ir)
  return ir
end

# Traces.
abstract type RecordSite end

struct Trace <: MixTable
    chm::Dict{Symbol, RecordSite}
    Trace() = new(Dict{Symbol, RecordSite}())
end

struct ChoiceSite{T} <: RecordSite
    score::Float64
    val::T
end

struct CallSite{T <: Trace, J, K} <: RecordSite
    trace::T
    fn::Function
    args::J
    ret::K
end

export remix!, MixTable

end # module
