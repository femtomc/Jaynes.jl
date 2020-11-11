const randprims = Set([rand, randn, randexp, randperm, shuffle, sample])
const blacklist = Set([Core.apply_type])
const primnames = Set(Symbol(fn) for fn in randprims)

"""
    Options{recurse::Bool, useslots::Bool, naming::Symbol}

Option type that specifies how Julia methods should be transformed into
generative functions. These options are passed as type parameters so
that they are accessible within `@generated` functions.
"""
struct Options{R, U, S} end
const MinimalOptions = Options{false, false, :static}
const DefaultOptions = Options{true, true, :static}

Options(recurse::Bool, useslots::Bool, naming::Symbol) =
    Options{recurse, useslots, naming}()

"Unpack option type parameters as a tuple."
unpack(::Options{R, U, S}) where {R, U, S} = (R, U, S)

"Transform the IR by wrapping sub-calls in `trace`(@ref)."
function automatic_addressing_transform!(ir::IR, options::Options=DefaultOptions())
    recurse, useslots, naming = unpack(options)
    # Modify arguments
    #optarg = argument!(ir; at=1) # Add argument for options
    rand_addrs = Dict() # Map from rand statements to address statements
    # Iterate over IR
    for (x, stmt) in ir
        !isexpr(stmt.expr, :call) && continue
        fn, args, calltype = unpack_call(stmt.expr)
        (unwrap(fn) == :trace || 
         unwrap(fn) == :learnable ||
         fn isa Variable ||
         !istraced(ir, fn, recurse)) && continue
        # Generate address name from function name and arguments
        addr = genaddr(ir, fn, unpack_args(ir, args, calltype))
        addr = insert!(ir, x, QuoteNode(addr))
        rand_addrs[x] = addr # Remember IRVar for address

        # Rewrite statement by wrapping call within `trace`
        rewrite!(ir, x, calltype, QuoteNode(options), addr, fn, args)
    end
    if (useslots) slotaddrs!(ir, rand_addrs) end # Name addresses using slots
    uniqueaddrs!(ir, rand_addrs) # Ensure uniqueness of random addresses
    loopaddrs!(ir, rand_addrs) # Add loop indices to addresses
    return ir
end

"Unpack calls, special casing `Core._apply` and `Core._apply_iterate`."
function unpack_call(expr::Expr)
    fn, args, calltype = expr.args[1], expr.args[2:end], :call
    if fn == GlobalRef(Core, :_apply)
        fn, args, calltype = args[1], args[2:end], :apply
    elseif fn == GlobalRef(Core, :_apply_iterate)
        fn, args, calltype = args[2], args[3:end], :apply_iterate
    end
    return fn, args, calltype
end

"Unpack tuples in IR."
unpack_tuple(ir, v::Variable) = haskey(ir, v) ?
unpack_tuple(ir, ir[v].expr) : nothing
unpack_tuple(ir, e) = iscall(e, GlobalRef(Core, :tuple)) ?
e.args[2:end] : iscall(e, GlobalRef(Base, :getindex)) ? e : error("Expected tuple, got $e.")

"Unpack arguments, special casing `Core._apply` and `Core._apply_iterate`."
function unpack_args(ir, args, calltype)
    if calltype == :call return args end
    unpacked = [unpack_tuple(ir, a) for a in args]
    filter!(a -> !isnothing(a), unpacked)
    return reduce(vcat, unpacked; init=[])
end

"Determine whether a called function should be traced."
function istraced(ir, fn::GlobalRef, recurse::Bool)
    if !isdefined(fn.mod, fn.name) return error("$fn not defined.") end
    val = getfield(fn.mod, fn.name)
    if val in randprims return true end # Primitives are always traced
    if !recurse return false end # Only trace primitives if not recursing
    for m in (Base, Core, Core.Intrinsics) # Filter out Base, Core, etc.
        if isdefined(m, fn.name) && getfield(m, fn.name) == val return false end
    end
    if val isa Type && val <: Sampleable return false end # Filter distributions
    val isa DataType && return false
    val isa UnionAll && return false
    return true
end
function istraced(ir, fn::Function, recurse::Bool) # Handle injected functions
    !(fn in blacklist) && (fn in randprims || recurse)
end

istraced(ir, fn::Variable, recurse::Bool) = # Handle IR variables
!haskey(ir, fn) || istraced(ir, ir[fn].expr, recurse)

istraced(ir, fn::Expr, recurse::Bool) = # Handle keyword functions
iscall(fn, GlobalRef(Core, :kwfunc)) ?
istraced(ir, fn.args[2], recurse) : true
istraced(ir, fn, recurse::Bool) = # Return true by default, to be safe
true

"Static generation of address names."
function genaddr(ir, fn::Symbol, args)
    if length(args) == 0 || !(fn in primnames) return fn end
    argsym = argaddr(ir, args[1])
    return isnothing(argsym) ? fn : Symbol(fn, :_, argsym)
end
function genaddr(ir, fn::Expr, args)
    if iscall(fn, GlobalRef(Core, :kwfunc))
        return genaddr(ir, fn.args[2], args[3:end])
    elseif isexpr(fn, :call) && fn.args[1] isa GlobalRef
        return genaddr(ir, Symbol(Expr(:call, fn.args[1].name)), args)
    else
        return :unknown
    end
end
genaddr(ir, fn::GlobalRef, args) =
genaddr(ir, fn.name, args)
genaddr(ir, fn::Function, args) =
genaddr(ir, nameof(fn), args)
genaddr(ir, fn::Variable, args) =
genaddr(ir, haskey(ir, fn) ? ir[fn].expr : argname(ir, fn), args)
genaddr(ir, fn, args) =
:unknown

"Generate partial address from argument."
argaddr(ir, arg::Variable) =
haskey(ir, arg) ? argaddr(ir, ir[arg].expr) : argname(ir, arg)
argaddr(ir, arg::Expr) =
isexpr(arg, :call) ? argaddr(ir, arg.args[1]) : nothing
argaddr(ir, arg::GlobalRef) =
arg.name
argaddr(ir, arg) =
Symbol(arg)

"Get argument slot name from IR."
argname(ir::IR, v::Variable) =
argname(ir.meta, v)
argname(meta::IRTools.Meta, v::Variable) =
1 <= v.id <= meta.nargs ? meta.code.slotnames[v.id] : nothing
argname(meta::Any, v::Variable) =
nothing

"Rewrites the statement by wrapping call within `trace`."
function rewrite!(ir, var, calltype, options, addr, fn, args)
    if calltype == :call # Handle basic calls
        if unwrap(fn) == :rand
            ir[var] = xcall(Jaynes, :trace, addr, args...)
        else
            ir[var] = xcall(Jaynes, :trace, addr, fn, args...)
        end
    elseif calltype == :apply # Handle `Core._apply`
        preargs = xcall(Core, :tuple, options, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(Core, :_apply,
                        GlobalRef(Jaynes, :trace), preargs, args...)
    elseif calltype == :apply_iterate # Handle `Core._apply_iterate`
        preargs = xcall(Core, :tuple, addr, fn)
        preargs = insert!(ir, var, preargs)
        ir[var] = xcall(Core, :_apply_iterate, GlobalRef(Base, :iterate),
                        GlobalRef(Jaynes, :trace), preargs, args...)
    end
end

"Generate trace addresses from slotnames where possible."
function slotaddrs!(ir::IR, rand_addrs::Dict)
    slotnames = ir.meta.code.slotnames
    for (x, stmt) in ir
        if isexpr(stmt.expr, :(=))
            # Attempt to automatically generate address names from slot names
            slot, v = stmt.expr.args
            if !isa(slot, IRTools.Slot) || !(v in keys(rand_addrs)) continue end
            slot_id = parse(Int, string(slot.id)[2:end])
            addr = slotnames[slot_id] # Look up name in CodeInfo
            ir[rand_addrs[v]] = QuoteNode(addr) # Replace previous address
        end
    end
    return ir
end

"Ensure that all addresses have unique names."
function uniqueaddrs!(ir::IR, rand_addrs::Dict)
    counts = Dict{Symbol,Int}()
    firstuses = Dict{Symbol,Variable}()
    addrvars = sort(collect(values(rand_addrs)), by = v -> ir.defs[v.id])
    for v in addrvars # Number all uses after first occurrence
        addr = ir[v].expr.value
        counts[addr] = get(counts, addr, 0) + 1
        if (counts[addr] == 1) firstuses[addr] = v; continue end
        ir[v] = QuoteNode(Symbol(addr, :_, counts[addr]))
    end
    for (addr, c) in counts # Go back and number first use of name
        if c == 1 continue end
        ir[firstuses[addr]] = QuoteNode(Symbol(addr, :_, 1))
    end
    return ir
end

"Add loop indices to addresses."
function loopaddrs!(ir::IR, rand_addrs::Dict)
    # Add count variables for each loop in IR
    loops, countvars = loopcounts!(ir)
    # Append loop count to addresses in each loop body
    for (loop, count) in zip(loops, countvars)
        for addrvar in values(rand_addrs)
            if !(block(ir, addrvar).id in loop.body) continue end
            if iscall(ir[addrvar].expr, GlobalRef(Base, :Pair))
                head, tail = ir[addrvar].expr.args[2:3]
                tail = insert!(ir, addrvar, xcall(Base, :Pair, tail, count))
                ir[addrvar] = xcall(Base, :Pair, head, tail)
            else
                head = insert!(ir, addrvar, ir[addrvar])
                ir[addrvar] = xcall(Base, :Pair, head, count)
            end
        end
    end
    return ir
end
