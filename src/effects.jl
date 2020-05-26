# These parallel the combinators of Gen. These effects are given a special semantics inside the TraceCtx in overdub.
function Chorus(call::Function, args)
    call(args...)
end

function Wavefolder(call::Function, args)
    call(args...)
end

# TODO: rewrite these. They both recurse, but they should really be efficient primitives.
function Cassette.overdub(ctx::TraceCtx,
                          call::typeof(rand),
                          addr::T,
                          m::typeof(Chorus),
                          args) where T <: Address
    isempty(args) && error("ChorusError: arguments are empty!")
    !(typeof(args[1]) <: Function) && error("ChorusError: first element of arguments tuple is not a function. Element type is $(typeof(args[1])).")
    call = args[1]
    func = () -> begin
        arr = PersistentVector([rand(addr => i, call, a) for (i, a) in enumerate(args[2])])
        return arr
    end
    ret = recurse(ctx, func)
    return ret
end

function Cassette.overdub(ctx::TraceCtx,
                          call::typeof(rand),
                          addr::T,
                          f::typeof(Wavefolder),
                          args) where T <: Address
    isempty(args) && error("LooperError: arguments are empty!")
    !(typeof(args[1]) <: Function) && error("LooperError: first element of arguments tuple is not a function. Element type is $(typeof(args[1])).")
    call = args[1]
    func = () -> begin
        iter = Int(args[2])
        state = args[3]
        for i in 1:iter
            state = rand(addr => i, call, state)
        end
        return state
    end
    ret = recurse(ctx, func)
    return ret
end
