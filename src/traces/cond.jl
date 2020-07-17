# ------------ Branch trace ------------ #

mutable struct BranchTrace{T <: RecordSite, B <: RecordSite} <: Trace
    cond::T
    branch::B
    params::Dict{Address, LearnableSite}
    BranchTrace(cond::T, branch::B) where {T, B} = new{T, B}(cond, branch, Dict{Address, LearnableSite}())
end

# If-else branch site
mutable struct ConditionalBranchCallSite{C, A, J, L, R} <: CallSite
    trace::BranchTrace
    score::Float64
    cond_kernel::C
    cond_args::J
    cond::Bool
    branch::A
    branch_args::L
    ret::R
end
get_score(cbs::ConditionalBranchCallSite) = cbs.score
