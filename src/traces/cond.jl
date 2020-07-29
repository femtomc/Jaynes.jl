# ------------ Branch trace ------------ #

struct BranchTrace{T <: RecordSite, B <: RecordSite} <: Trace
    cond::T
    branch::B
    BranchTrace(cond::T, branch::B) where {T, B} = new{T, B}(cond, branch)
end

# If-else branch site
struct ConditionalBranchCallSite{C, A, J, L, R} <: CallSite
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
