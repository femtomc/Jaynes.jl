# ------------ Abstract interpreters ------------ #

abstract type InterpretationContext end

# Fallback.
#absint(args...) = Union{}

# Convenience definitions of abstract interpretations.
closure_check(e) = isexpr(e, :(::))
function _abstract(ctx, expr)
    expr = longdef(expr)
    if @capture(expr, function name_(args__) body_ end)
        !closure_check(name) ? decl = Expr(:(::), :fn, 
                                           Expr(:call, :typeof, name)) : decl = name
        new = quote
            function absint(ctx::$ctx, $decl, $(args...))
                $body
            end
        end
    elseif @capture(expr, function name_(args__) where {T__} body_ end)
        !closure_check(name) ? decl = Expr(:(::), :fn, 
                                           Expr(:call, :typeof, name)) : decl = name
        new = quote
            function absint(ctx::$ctx, $decl, $(args...)) where {$(T...)}
                $body
            end
        end
    else
        error("@abstract: unable to parse abstract interpretation definition.")
    end
    return MacroTools.postwalk(unblock ∘ rmlines, new)
end

macro abstract(ctx, expr)
    new = _abstract(ctx, expr)
    esc(new)
end

resolve(x) = x
resolve(gr::GlobalRef) = getproperty(gr.mod, gr.name)

# ------------ Abstract interpretation pipeline ------------ #

function prepare_ir!(ir)
    for (v, st) in ir
        isexpr(st.expr) || continue
        ir[v] = stmt(Expr(st.expr.head, map(resolve, st.expr.args)...); type = Union{})
    end
    ir
end

function infercall!(ctx::InterpretationContext, env, v, st, ir)
    args = map(st.expr.args) do a
        k = unwrap(a)
        get(env, k, k)
    end
    t = absint(ctx, args...)
    env[v] = t
    ir[v] = stmt(st.expr; type = t)
end

function infer!(ctx::InterpretationContext, env::Dict, ir)
    prepare_ir!(ir)
    for (v, st) in ir
        isexpr(st.expr, :call) && infercall!(ctx, env, v, st, ir)
    end
    ir
end
@inline infer!(ctx::InterpretationContext, ir) = infer!(ctx, Dict(), ir)

function infer!(ctx::InterpretationContext, fn::Function, argtypes...)
    ir = lower_to_ir(fn, argtypes...)
    infer!(ctx, ir)
end
