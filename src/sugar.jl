function _sugar(expr)
    MacroTools.postwalk(expr) do s
        if @capture(s, val_ ~ d_)

            # Matches: x ~ distribution.
            if val isa QuoteNode
                k = quote rand($val, $d) end

                # Matches: x = (:x => 5) ~ distribution
            elseif val isa Expr
                k = quote rand($val, $d) end

                # Matches: x = (:x) ~ distribution
            else
                addr = QuoteNode(val)
                k = quote $val = rand($addr, $d) end
            end

            # Fallthrough.
        else
            k = s
        end
        (unblock ∘ rmlines)(k)
    end
end

macro sugar(expr)
    trans = MacroTools.postwalk(unblock ∘ rmlines, _sugar(expr))
    trans
end
