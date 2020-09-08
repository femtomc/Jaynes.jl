# Get all distributions in Distributions.jl
distributions = map(subtypes(Distribution)) do t
    Symbol(t)
end

function _sugar(expr)
    MacroTools.postwalk(expr) do s
        if @capture(s, {addr_} ~ fn_(args__))
            if Symbol("Distributions.$fn") in distributions || fn in distributions
                k = Expr(:call, :rand, addr, Expr(:call, fn, args...))
            else
                k = Expr(:call, :rand, addr, fn, args...)
            end

        elseif @capture(s, val_ ~ fn_(args__))
            val isa Expr && error("Raw value assignment ~ for choice requires that value be a variable name (e.g. x, y, z, ...).")
            addr = QuoteNode(val)
            if Symbol("Distributions.$fn") in distributions || fn in distributions
                k = Expr(:(=), val, Expr(:call, :rand, addr, Expr(:call, fn, args...)))
            else
                k = Expr(:(=), val, Expr(:call, :rand, addr, fn, args...))
            end

        elseif @capture(s, val_ <- fn_(args__))
            k = quote $val = deep($fn, $(args...)) end

        else
            # Fallthrough.
            k = s
        end
        (unblock ∘ rmlines)(k)
    end
end

macro sugar(expr)
    trans = MacroTools.postwalk(unblock ∘ rmlines, _sugar(expr))
    esc(trans)
end
