abstract type KernelDSL end
struct Basic <: KernelDSL end
struct Involution <: KernelDSL end

struct MCMCMeta{T <: KernelDSL}
    dsl::T
end

function Cassette.overdub(ctx::TraceCtx{M}, fn::Function, args) where M <: MCMCMeta
    error("$fn not allowed in the $(ctx.metadata.dsl) domain-specific language.")
end

function kernel_step!(K::KernelDSL, 
                      tr::Trace, 
                      select::UnconstrainedSelection, 
                      fn::Function, 
                      args::Tuple)
    tr = fn(tr, select, args...)
    return tr
end

kernel_step!(tr::Trace, select::UnconstrainedSelection, fn::Function, args::Tuple) = kernel_step!(Basic(), tr, select, fn, args)
kernel_step!(tr::Trace, select::UnconstrainedSelection, fn::Function, args::Tuple; K::KernelDSL) = kernel_step!(K, tr, select, fn, args)

# Simple primitives for a simple collection!
@add_kernel! Basic function rand(tr::Trace, sel::UnconstrainedSelection)::Trace
end
@add_kernel! Basic function metropolis_hastings(tr::Trace, sel::UnconstrainedSelection)::Trace
end
