# ---- None of these are working. ---- #

get_mod(gr::GlobalRef) = gr.mod
get_mod(gr) = gr

unwrap(gr::GlobalRef) = gr.name
unwrap(gr) = gr

function check_nooverdub(k::Expr)
    check = !isempty(k.args) && k.args[1] isa Expr && k.head == :nooverdub
    check
end

function insert_nooverdub(k::Expr)
    len = length(k.args)
    len == 1 && return Expr(:call, Expr(:nooverdub, k.args[1]))
    out = Expr(:call, Expr(:nooverdub, k.args[1]), k.args[2:end]...)
    Core.println(out)
    out
end

match_nooverdub(k) = false
function match_nooverdub(k::Expr)
    check_nooverdub(k) && return false

    # Check for rand call.
    if k.head == :call
        unwrap(k.args[1]) == :rand && begin
            return false
        end
        get_mod(k.args[1]) in [Core, Base] && return true
    end
    return false
end

function ignore_transform!(::Type{<:TraceCtx}, r::Reflection)
    for k in r.code_info.code
        Cassette.replace_match!(insert_nooverdub, match_nooverdub, k)
    end
    return r.code_info
end

function ignore_transform2!(::Type{<:TraceCtx}, r::Reflection)
    syn = r.code_info.code
    map(syn) do expr
        MacroTools.postwalk(expr) do k
            # If you already wrapped, don't wrap.
            k isa Expr && k.head == :call && begin
                arg = k.args[1]
                arg isa Expr && arg.head == :nooverdub && return k
            end

            # If you haven't wrapped, wrap.
            k isa Expr && k.head == :call && begin
                call = k.args[1]
                if !(call isa GlobalRef && call.name == :rand)
                    k.args[1] = Expr(:nooverdub, call)
                    return k
                end
            end
            
            k
        end
    end
    return r.code_info
end

# ----- Version from KernelAbstractions.jl works. ----- #

function ir_element(x, code::Vector)
    while isa(x, Core.SSAValue)
        x = code[x.id]
    end
    return x
end

function transform!(ctx, ref)
    CI = ref.code_info

    # don't overdub pure functions
    if CI.pure
        n_method_args = Int(ref.method.nargs)
        if ref.method.isva
            Cassette.insert_statements!(CI.code, CI.codelocs,
                                        (x, i) -> i == 1 ?  3 : nothing,
                                        (x, i) -> i == 1 ? [
                                                            # this could run into troubles when the function is @pure f(x...) since then n_method_args==2, but this seems to work sofar.
                                                            Expr(:call, Expr(:nooverdub, GlobalRef(Core, :tuple)), (Core.SlotNumber(i) for i in 2:(n_method_args-1))...),
                                                            Expr(:call, Expr(:nooverdub, GlobalRef(Core, :_apply)), Core.SlotNumber(1), Core.SSAValue(i), Core.SlotNumber(n_method_args)),
                                                            Expr(:return, Core.SSAValue(i+1))] : nothing)
        else
            Cassette.insert_statements!(CI.code, CI.codelocs,
                                        (x, i) -> i == 1 ?  2 : nothing,
                                        (x, i) -> i == 1 ? [
                                                            Expr(:call, Expr(:nooverdub, Core.SlotNumber(1)), (Core.SlotNumber(i) for i in 2:n_method_args)...)
                                                            Expr(:return, Core.SSAValue(i))] : nothing)
        end
        CI.ssavaluetypes = length(CI.code)
        return CI
    end

    # overdubbing IntrinsicFunctions removes our ability to profile code
    newstmt = (x, i) -> begin
        isassign = Base.Meta.isexpr(x, :(=))
        stmt = isassign ? x.args[2] : x
        if Base.Meta.isexpr(stmt, :call)
            applycall = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply), CI.code) 
            applyitercall = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply_iterate), CI.code) 
            if applycall
                fidx = 2
            elseif applyitercall
                fidx = 3
            else
                fidx = 1
            end
            f = stmt.args[fidx]
            f = ir_element(f, CI.code)
            if f isa GlobalRef
                mod = f.mod
                name = f.name
                if Base.isbindingresolved(mod, name) && Base.isdefined(mod, name)
                    ff = getfield(f.mod, f.name)
                    if ff isa Core.IntrinsicFunction || ff isa Core.Builtin
                        stmt.args[fidx] = Expr(:nooverdub, f)
                    end
                end
            end
        end
        return [x]
    end

    Cassette.insert_statements!(CI.code, CI.codelocs, (x, i) -> 1, newstmt)
    CI.ssavaluetypes = length(CI.code)
    # Core.Compiler.validate_code(CI)
    return CI
end

const ignore_pass = Cassette.@pass transform!
