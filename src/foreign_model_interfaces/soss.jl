macro load_soss_fmi()
    expr = quote
        using Soss

        mutable struct SossCallSite{M, F} <: CallSite
            model::M
            logpdf::F
            logprob::Float64
            args::Tuple
        end

        function (ctx::GenerateContext)(c::typeof(soss_fmi),
                                        fn::typeof(rand),
                                        addr::Address,
                                        call,
                                        args...)
        end
    end
    expr
end
