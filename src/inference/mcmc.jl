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

kernel_step!(fn::Function, args::Tuple, tr::Trace, kernel::Function, k_args::Tuple) = kernel_step!(Basic(), tr, select, fn, args)
kernel_step!(regen_ctx::TraceCtx{M}, tr::Trace, select::UnconstrainedSelection, kernel::Function, k_args::Tuple; K::KernelDSL = Basic()) where M <: RegenerateMeta = kernel_step!(K, tr, select, fn, args)

@add_kernel! Basic function metropolis_hastings(regen_ctx::TraceCtx{M}, tr::Trace, sel::UnconstrainedSelection)::Trace where M <: RegenerateMeta
end

function check(ir)
end

function reverse(ir)
end
