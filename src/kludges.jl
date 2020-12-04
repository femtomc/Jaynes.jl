# ------------ Com-pirate fixes ------------ #

# In the future, I hope to remove some of these.

import Zygote.literal_getproperty

# Fix for: https://github.com/FluxML/Zygote.jl/issues/717
Zygote.@adjoint function literal_getproperty(x, ::Val{f}) where f
    val = getproperty(x, f)
    function back(Δ)
        Zygote.accum_param(__context__, val, Δ) # === nothing && return
        if isimmutable(x)
            ((;Zygote.nt_nothing(x)..., Zygote.pair(Val(f), Δ)...), nothing)
        else
            dx = Zygote.grad_mut(__context__, x)
            dx[] = (;dx[]...,Zygote.pair(Val(f), Zygote.accum(getfield(dx[], f), Δ))...)
            return (dx, nothing)
        end
    end
    unwrap(val), back
end

# Whitelist includes vectorized calls. (Required for performance of tracing).
whitelist = [
             # Jaynes custom indicator call.
             :trace,

             # Base.
             :map, :filter, :_apply_iterate, :collect,

             # Interactions with the context.
             :learnable, :fillable, :factor,
            ]

# Fix for specialized tracing. (Not a big deal).
function recur(ir, to = self)
    pr = Pipe(ir)
    for (x, st) in pr
        isexpr(st.expr, :call) && begin
            ref = unwrap(st.expr.args[1])
            ref in whitelist || continue
            pr[x] = Expr(:call, to, st.expr.args...)
        end
    end
    finish(pr)
end

# Fix for _apply_iterate (used in contexts). (More annoying).
function f_push!(arr::Array, t::Tuple{}) end
f_push!(arr::Array, t::Array) = append!(arr, t)
f_push!(arr::Array, t::Tuple) = append!(arr, t)
f_push!(arr, t) = push!(arr, t)
function flatten(t::Tuple)
    arr = Any[]
    for sub in t
        f_push!(arr, sub)
    end
    return arr
end
