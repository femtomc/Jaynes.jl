function _jaynes(def, opt)

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
            JFunction($opt(),
                      $decl, 
                      tuple($argtypes...), 
                      tuple([false for _ in $argtypes]...), 
                      false, 
                      Any)
        end

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

    trans
end

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
