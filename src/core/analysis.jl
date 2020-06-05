world_counter() = ccall(:jl_get_world_counter, UInt, ())

# Recurse through the IR of each method invocation, for a particular set of arguments - this produces an Analysis object, which can be used by any of the contexts.
struct Analysis
    callgraph
    cacheable
end

function toplevel_analyze(fn::Function, args::Tuple)
    sig = typeof((fn, args...))
    m = meta(sig)
    return m.code
end
