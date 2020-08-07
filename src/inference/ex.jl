function exchange(sel::K, cl::C, addr::T, ker) where {K <: UnconstrainedSelection, C <: CallSite, T <: Tuple}
    target = getindex(cl, addr)
    new, acc = apply_kernel(sel, ker, target)
    sel = selection(addr => get_selection(new))
    ret, new, _ = update(sel, cl)
    new, true
end

apply_kernel(sel, ker, cl::HierarchicalCallSite) = ker(sel, cl)
apply_kernel(sel, ker, cl::VectorizedCallSite) = ker(sel, cl)
