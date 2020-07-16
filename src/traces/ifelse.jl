# ------------ Branch trace ------------ #

mutable struct BranchTrace{T <: RecordSite, B <: RecordSite} <: Trace
    condtrace::T
    branchtrace::B
    params::Dict{Address, LearnableSite}
end

# If-else branch site
mutable struct ConditionalBranchSite{C, A, B, T <: RecordSite, K <: RecordSite, J, L, R}
    trace::BranchTrace
    score::Float64
    cond_kernel::C
    cond_args::J
    a::A
    b::B
    branch_args::L
    ret::R
end
