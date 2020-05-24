# These parallel the combinators of Gen. These effects are given a special semantics in using contexts in overdub.

function chorus(call::Function) where T
    call
end

function delay(call::Function, iter::Int) where T
    call
end

function Cassette.overdub(ctx::TraceCtx,
                          call::typeof(rand),
                          addr::T,
                          call::typeof(chorus),
                          args) where T <: Address
end

function Cassette.overdub(ctx::TraceCtx,
                          call::typeof(rand),
                          addr::T,
                          call::typeof(delay),
                          args) where T <: Address
end
