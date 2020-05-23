module Timing

using Cassette
using Cassette: similarcontext
using BenchmarkTools

Cassette.@context IdentityCtx;

Cassette.overdub(::IdentityCtx, call::Function, args) = call(args...)

function foo()
    return 5 + 10
end

function bar(num_samples)
    ctx = IdentityCtx()
    for i in 1:num_samples
        Cassette.overdub(ctx, foo)
        ctx = similarcontext(ctx)
    end
end

@btime bar(50000)

end # module
