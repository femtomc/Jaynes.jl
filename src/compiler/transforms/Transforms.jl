abstract type Transform end

include("trace_transform/TraceTransform.jl")
include("logpdf_transform/LogPDFTransform.jl")
include("proposal_transform/ProposalTransform.jl")
include("static_graph_transform/StaticGraphTransform.jl")

macro transform(transform, expr)
    @capture(expr, f_(args__))
    quote
        # Normal func calls
        ir = @code_ir $transform $(esc(f))($(esc(args))...)
        func(ir)
    end
end
