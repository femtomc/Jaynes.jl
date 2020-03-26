Cassette.@context TraceCtx;

# -- Compiler pass  for lowered code -- #

# Utility function for inserting arguments into function calls in lowered code.
insert_addr_in_call = (expr, addr) -> postwalk(x -> @capture(x, f_(xs__)) ? (f isa GlobalRef && f.name == :rand ? :($f($addr, $(xs...))) : x) : x, expr)

function transform_sample_statements(ci::Core.CodeInfo)
    lowered = copy(ci)
    code = lowered.code

    # Mapping the code...
    identifier_dict = Dict(map(x -> (Core.SlotNumber(x[1]) => x[2]), enumerate(lowered.slotnames)))
    SSA_dict = Dict(map(x -> (Core.SSAValue(x[1]) => x[2]), enumerate(code)))

    # Insertion.
    transformed = map(line -> (line isa Expr && line.args[1] isa Core.SlotNumber) ? insert_addr_in_call(line, String(identifier_dict[line.args[1]])) : line, code)

    # Dependency insertion.
    #transformed = map(expr -> postwalk(y -> (y isa Core.SSAValue && @capture(SSA_dict[y], f_(xs__)) && (f isa GlobalRef && f.name == :rand)) ? insert_addr_in_call(transformed[y.id], y) : y, expr), transformed)

    # Insert new code.
    lowered.code = transformed
    return lowered
end

function insert_semantic_identifiers(::Type{<:TraceCtx}, reflection::Cassette.Reflection)
    lowered = reflection.code_info
    transformed = transform_sample_statements(lowered)
    return transformed
end

const insert_semantic_identifiers_pass = Cassette.@pass insert_semantic_identifiers 
# -- END COMPILER PASS --

function Cassette.overdub(ctx::TraceCtx, 
                 call::typeof(rand), addr::String, d::T) where T <: Distribution
    result = call(d)
    score = logpdf(d, result)
    record!(ctx.metadata, Symbol(addr), d, result, score)
    return result
end

function Cassette.overdub(ctx::TraceCtx, 
                 call::typeof(rand), d::T) where T <: Distribution
    result = call(d)
    score = logpdf(d, result)
    record!(ctx.metadata, gensym(), d, result, score)
    return result
end

macro trace(func, args...)
    quote
        tr = Trace($(esc(func)), $(esc(args))...)
        retval = Cassette.overdub(Cassette.disablehooks(TraceCtx(pass = insert_semantic_identifiers_pass, metadata = tr)), $(esc(func)), $(esc(args))...)
        tr.retval = retval
        tr
    end
end

