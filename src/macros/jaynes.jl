# Get all distributions in Distributions.jl
distributions = map(subtypes(Distribution)) do t
    Symbol(t)
end
push!(distributions, :Mixture)
push!(distributions, :Product)

# Desugar "sugared" tilde expressions.
function _sugar(expr)
    MacroTools.postwalk(expr) do s
        if @capture(s, {addr_} ~ fn_(args__))
            if Symbol("Distributions.$fn") in distributions || fn in distributions
                k = Expr(:call, :trace, addr, Expr(:call, fn, args...))
            else
                k = Expr(:call, :trace, addr, fn, args...)
            end

        elseif @capture(s, val_ ~ fn_(args__))
            val isa Expr && error("Raw value assignment ~ for choice requires that value be a variable name (e.g. x, y, z, ...).")
            addr = QuoteNode(val)
            if Symbol("Distributions.$fn") in distributions || fn in distributions
                k = Expr(:(=), val, Expr(:call, :trace, addr, Expr(:call, fn, args...)))
            else
                k = Expr(:(=), val, Expr(:call, :trace, addr, fn, args...))
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

# Core Jaynes parser.
function _jaynes(def, opt)

    # Matches longdef function definitions.
    R = Any
    if @capture(def, function decl_(args__) body__ end) || @capture(def, function decl_(args__)::R_ body__ end)
        argtypes = Expr(:tuple, map(args) do a
            if a isa Expr 
                a.head == :(::) ? a.args[2] : :Any
            else
                :Any
            end
        end...)
        trans = quote 
            $def
            argtypes = $argtypes
            JFunction($opt(),
                      $decl, 
                      argtypes,
                      tuple([false for _ in argtypes]...), 
                      false, 
                      $R)
        end
        trans

        # Matches thunks.
    elseif @capture(def, () -> body__)
        trans = quote
            JFunction($opt(),
                      $def, 
                      (Tuple{}, ),
                      (false, ),
                      false, 
                      Any)
        end

        # Matches lambdas with formals.
    elseif @capture(def, (args_) -> body__)
        if args isa Symbol
            argtypes = [Any]
        elseif args.head == :(::)
            argtypes = [eval(args.args[2])]
        else
            argtypes = map(args.args) do a
                if a isa Expr 
                    a.head == :(::) ? eval(a.args[2]) : Any
                else
                    Any
                end
            end
        end
        trans = quote
            JFunction($opt(),
                      $def, 
                      tuple($argtypes...), 
                      tuple([false for _ in $argtypes]...), 
                      false, 
                      Any)
        end
    else
        error("ParseError (@jaynes): requires a longdef function definition or an anonymous function definition.")
    end

    MacroTools.postwalk(unblock ∘ rmlines, trans)
end

# @jaynes macro.
macro jaynes(expr, opt)
    def = _sugar(expr)
    trans = _jaynes(def, opt)
    esc(trans)
end

macro jaynes(expr)
    def = _sugar(expr)
    trans = _jaynes(def, :DefaultPipeline)
    esc(trans)
end
