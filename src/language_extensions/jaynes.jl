function _jaynes(check, hints, def)

    # Matches longdef function definitions.
    if @capture(def, function decl_(args__) body__ end)
        argtypes = map(args) do a
            if a isa Expr 
                a.head == :(::) ? eval(a.args[2]) : Any
            else
                Any
            end
        end
        trans = quote 
            $def
            JFunction($decl, 
                      tuple($argtypes...), 
                      tuple([false for _ in $argtypes]...), 
                      false, 
                      Any;
                      static_checks = $check,
                      hints = $hints)
        end

        # Matches thunks.
    elseif @capture(def, () -> body__)
        trans = quote
            JFunction($def, 
                      (Tuple{}, ),
                      (false, ),
                      false, 
                      Any;
                      static_checks = $check,
                      hints = $hints)
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
            JFunction($def, 
                      tuple($argtypes...), 
                      tuple([false for _ in $argtypes]...), 
                      false, 
                      Any;
                      static_checks = $check,
                      hints = $hints)
        end
    else
        error("ParseError (@jaynes): requires a longdef function definition or an anonymous function definition.")
    end

    trans
end

macro jaynes(expr)
    def = _sugar(expr)
    trans = _jaynes(false, false, def)
    esc(trans)
end

macro jaynes(expr, flag)
    def = _sugar(expr)
    options = [:check, :hints]
    if flag isa Expr && flag.head == :tuple
        trans = _jaynes(map(options) do o
                            o in flag.args
                        end..., def)
    elseif flag == :check
        trans = _jaynes(true, false, def)
    elseif flag == :hints
        trans = _jaynes(false, true, def)
    end
    esc(trans)
end
