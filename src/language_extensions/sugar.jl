distributions = [:MvNormal,
                 :Normal,
                 :Categorical,
                 :Bernoulli,
                 :Dirichlet,
                 :InverseGamma]

function _sugar(expr)
    MacroTools.postwalk(expr) do s
        if @capture(s, val_ ~ fn_(args__))
            if fn in distributions
                if val isa QuoteNode
                    k = Expr(:call, :rand, val, Expr(:call, fn, args...))

                    # Matches: x = (:x => 5) ~ distribution
                elseif val isa Expr
                    k = Expr(:call, :rand, val, Expr(:call, fn, args...))

                    # Matches: x = (:x) ~ distribution
                else
                    addr = QuoteNode(val)
                    k = Expr(:(=), val, Expr(:call, :rand, addr, Expr(:call, fn, args...)))
                end

            else
                # Matches: x ~ fn(args...)
                if val isa QuoteNode
                    k = Expr(:call, :rand, val, fn, args...)

                    # Matches: x = (:x => 5) ~ fn(args...)
                elseif val isa Expr
                    k = Expr(:call, :rand, val, fn, args...)

                    # Matches: x = (:x) ~ fn(args...)
                else
                    addr = QuoteNode(val)
                    k = Expr(:(=), val, Expr(:call, :rand, addr, fn, args...))
                end
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
