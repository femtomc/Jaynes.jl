# ------------ Dynamic addressing hint ------------ #

# Provides a warning to users when using dynamic addressing.

struct DynamicAddressingHint <: AddressingHint
    dynamic_addresses::Vector{Variable}
    ir_map::Dict{Variable, IRTools.Statement}
end

@inline Base.isempty(dah::DynamicAddressingHint) = isempty(dah.dynamic_addresses)

function Base.display(dah::DynamicAddressingHint)
    if isempty(dah)
        println("\u001b[32m ✓ (DynamicAddressHint): Compiler detected no dynamic addresses in your model code.\u001b[0m")
    else
        println("\u001b[33m (DynamicAddressHint): Compiler detected the following dynamic addresses in your model code.\u001b[0m")
        println("________________________\n")
        display(dah.ir_map)
        println("________________________\n")
        println("\u001b[31m (Warning): This will prevent the compiler from inferring a trace type for your model program, as well as statically checking the support.\u001b[0m")
        println("\u001b[32m (Recommendation): transform control flow constructs into \u001b[3m\u001b[34;1mcombinators\u001b[0m\n\n      \u001b[34;1mhttps://www.gen.dev/dev/ref/combinators/\u001b[0m\n\n\u001b[32mThen, provide static addresses to combinator calls.\u001b[0m")
    end
end

function detect_dynamic_addresses(ir)
    dynamic_addresses = Variable[]
    ir_map = Dict{Variable, IRTools.Statement}()
    for (v, st) in ir
        st.expr isa Expr || continue
        st.expr.head == :call || continue
        (st.expr.args[1] == trace || unwrap(st.expr.args[1]) == :trace) || continue
        st.expr.args[2] isa QuoteNode && continue
        addr = st.expr.args[2]
        if addr isa Variable
            st = ir[addr]
            vars = Set{Variable}([])
            if st.expr isa Expr
                MacroTools.postwalk(st.expr) do e
                    e isa Variable && push!(vars, addr)
                    e
                end
            end
            ir_map[addr] = st
            append!(dynamic_addresses, vars)
        end
    end
    dh = DynamicAddressingHint(dynamic_addresses, ir_map)
    dh
end

function detect_dynamic_addresses(func, arg_types...)
    ir = lower_to_ir(func, arg_types...)
    display(detect_dynamic_addresses(ir))
end

@inline dynamic_address_check(ir) = !isempty(detect_dynamic_addresses(ir))
