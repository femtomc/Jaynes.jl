# ------------ DSL implementation for Jaynes ------------ #

# Utilities.
is_choice(::Choice) = true
is_choice(a) = false
is_value(::Value) = true
is_value(a) = false

# ------------ Selections ------------ #

struct JSelection{K <: AddressMap{Select}} <: Selection
    sel::K
end
unwrap(js::JSelection) = js.sel

Base.in(addr, selection::JSelection) = haskey(selection.sel, addr)
Base.getindex(selection::JSelection, addr) = get_sub(selection.sel, addr)
Base.isempty(selection::JSelection, addr) = isempty(selection.sel, addr)

select(v::Vector{T}) where T <: Tuple = JSelection(target(v))
display(js::JSelection) = display(unwrap(js))

# ------------ Choice map ------------ #

struct JChoiceMap{K <: AddressMap} <: ChoiceMap
    chm::K
end
unwrap(jcm::JChoiceMap) = jcm.chm

has_value(choices::JChoiceMap, addr) = has_value(unwrap(choices), addr)
get_value(choices::JChoiceMap, addr) = getindex(unwrap(choices), addr)
get_submap(choices::JChoiceMap, addr) = get_sub(unwrap(choices), addr)

# TODO: fix.
function get_values_shallow(choices::JChoiceMap)
    shallow_iterator(unwrap(choices))
end
function get_submaps_shallow(choices::JChoiceMap)
    shallow_iterator(unwrap(choices))
end

function merge(chm1::JChoiceMap, chm2::JChoiceMap)
    new, check = merge(unwrap(chm1), unwrap(chm2))
    JChoiceMap(new)
end

to_array(choices::JChoiceMap, ::Type{T}) where T = array(unwrap(choices), T)
from_array(choices::JChoiceMap, arr::Vector) = target(unwrap(choices), arr)
display(jcm::JChoiceMap) = Jaynes.display(unwrap(jcm))

choicemap(c::Vector{Pair{T, K}}) where {T <: Tuple, K} = JChoiceMap(target(c))
function convert(::Type{DynamicMap{Value}}, chm::DynamicChoiceMap)
    dm = DynamicMap{Value}()
    for (k, v) in get_values_shallow(chm)
        set_sub!(dm, k, Value(v))
    end
    for (k, v) in get_submaps_shallow(chm)
        sub = convert(DynamicMap{Value}, v)
        set_sub!(dm, k, sub)
    end
    dm
end

function convert(::Type{DynamicMap{Select}}, sel::Gen.DynamicSelection)
    dm = DynamicMap{Select}()
    for (k, v) in Gen.get_subselections(sel)
        if v isa Gen.AllSelection
            set_sub!(dm, k, SelectAll())
        else
            sub = convert(DynamicMap{Select}, v)
            set_sub!(dm, k, sub)
        end
    end
    dm
end

static(chm::DynamicChoiceMap) = static(convert(DynamicMap{Value}, chm))

# ------------ Trace ------------ #

mutable struct JTrace{T, K <: CallSite} <: Trace
    gen_fn::T
    record::K
    isempty::Bool
end

@inline get_record(trace::JTrace) = trace.record
@inline set_retval!(trace::JTrace, retval) = (trace.retval = retval)
@inline has_choice(trace::JTrace, addr) = haskey(trace.record, addr) && is_choice(get_sub(trace.record, addr))
@inline has_value(trace::JTrace, addr) = has_value(get_record(trace), addr)
@inline get_value(trace::JTrace, addr) = getindex(get_record(trace), addr)

function get_choice(trace::JTrace, addr)
    ch = get_sub(trace.record, addr)
    !is_choice(ch) && throw(KeyError(addr))
    ch
end

get_choices(trace::JTrace) = get_trace(trace.record)

Base.display(jtr::JTrace) = Base.display(get_trace(jtr.record))

# Trace GFI methods.
get_args(trace::JTrace) = get_args(trace.record)
get_retval(trace::JTrace) = get_ret(trace.record)
get_score(trace::JTrace) = get_score(trace.record)
get_gen_fn(trace::JTrace) = trace.gen_fn

# ------------ Generative function ------------ #

struct JFunction{N, R} <: GenerativeFunction{R, JTrace}
    fn::Function
    params::DynamicMap{Value}
    params_grads::DynamicMap{Value}
    arg_types::NTuple{N, Type}
    has_argument_grads::NTuple{N, Bool}
    accepts_output_grad::Bool
    ir::IR
    reachability::FlowAnalysis
end

function JFunction(func::Function,
                   arg_types::NTuple{N, Type},
                   has_argument_grads::NTuple{N, Bool},
                   accepts_output_grad::Bool,
                   ::Type{R}) where {N, R}
    ir = lower_to_ir(func, arg_types...)
    JFunction{N, R}(func, 
                    DynamicMap{Value}(), 
                    DynamicMap{Value}(), 
                    arg_types, 
                    has_argument_grads, 
                    accepts_output_grad,
                    ir,
                    flow_analysis(ir))
end

@inline (jfn::JFunction)(args...) = jfn.fn(args...)

@inline has_argument_grads(jfn::JFunction) = jfn.has_argument_grads

@inline get_params(jfn::JFunction) = jfn.params
@inline get_params_grads(jfn::JFunction) = jfn.params_grads

init_param!(jfn, addr, val) = set_sub!(jfn.params, addr, Value(val))
init_param!(jfn, v::Vector{Pair{T, K}}) where {T <: Tuple, K} = begin
    for (addr, val) in v
        init_param!(jfn, addr, val)
    end
end

@inline get_analysis(jfn::JFunction) = jfn.reachability

@inline get_ir(jfn::JFunction) = jfn.ir

function generate_graph_ir(jfunc::JFunction)
    ssa_ir = get_ir(jfunc)
    flow = get_analysis(jfunc)
    graph_ir = graph_walk(ssa_ir, flow)
    graph_ir
end

# ------------ Model GFI interface ------------ #

function simulate(jfn::JFunction, args::Tuple)
    ret, cl = simulate(get_params(jfn), 
                       jfn.fn, 
                       args...)
    JTrace(jfn, cl, false)
end

function generate(jfn::JFunction, args::Tuple, chm::JChoiceMap)
    ret, cl, w = generate(unwrap(chm), 
                          get_params(jfn), 
                          jfn.fn, 
                          args...)
    JTrace(jfn, cl, false), w
end
@inline generate(jfn::JFunction, args::Tuple, choices::DynamicChoiceMap) = generate(jfn, args, JChoiceMap(convert(DynamicMap{Value}, choices)))
@inline generate(jfn::JFunction, args::Tuple, choices::DynamicMap) = generate(jfn, args, JChoiceMap(choices))

function assess(jfn::JFunction, args::Tuple, choices::JChoiceMap)
    ret, w = assess(unwrap(choices), 
                   get_params(jfn), 
                   jfn.fn, 
                   args...)
    w, ret
end
@inline assess(jfn::JFunction, args::Tuple, choices::DynamicChoiceMap) = assess(jfn, args, JChoiceMap(convert(DynamicMap{Value}, choices)))

function project(jtr::JTrace, sel::JSelection)
    w, proj = projection(get_record(jtr), unwrap(sel))
    w
end

function project(jtr::JTrace, sel::EmptySelection)
    w, proj = projection(get_record(jtr), Empty())
    w
end

function propose(jfn::JFunction, args::Tuple)
    ret, chm, w = propose(get_params(jfn), 
                          jfn.fn, 
                          args...)
    JChoiceMap(chm), w, ret
end

function update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::JChoiceMap)
    ret, cl, w, rd, d = update(unwrap(constraints), 
                               get_params(get_gen_fn(trace)), 
                               trace.record, 
                               map(zip(args, arg_diffs)) do (a, d)
                                   Diffed(a, d)
                               end...)
    JTrace(get_gen_fn(trace), cl, false), w, rd, JChoiceMap(d)
end
@inline update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::DynamicChoiceMap) = update(trace, args, arg_diffs, JChoiceMap(convert(DynamicMap{Value}, constraints)))
@inline update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::StaticMap) = update(trace, args, arg_diffs, JChoiceMap(constraints))

function regenerate(trace::JTrace, args::Tuple, arg_diffs::Tuple, selection::JSelection)
    ret, cl, w, rd, d = regenerate(unwrap(selection), 
                               get_params(get_gen_fn(trace)), 
                               trace.record, 
                               map(zip(args, arg_diffs)) do (a, d)
                                   Diffed(a, d)
                               end...)
    JTrace(get_gen_fn(trace), cl, false), w, rd, JChoiceMap(d)
end
@inline regenerate(trace::JTrace, args::Tuple, arg_diffs::Tuple, selection::Gen.DynamicSelection) = regenerate(trace, args, arg_diffs, JSelection(convert(DynamicMap{Select}, selection)))
@inline regenerate(trace::JTrace, args::Tuple, arg_diffs::Tuple, selection::StaticMap) = regenerate(trace, args, arg_diffs, JSelection(selection))

# ------------ Gradients ------------ #

function choice_gradients(tr::JTrace, 
                          selection::JSelection = JSelection(SelectAll()), 
                          retgrad = nothing)
    vals, as, cgs = get_choice_gradients(unwrap(selection), 
                                         get_params(get_gen_fn(tr)), 
                                         get_record(tr), 
                                         retgrad)
    as, vals, cgs
end

function accumulate_param_gradients!(tr::JTrace, retgrad = (nothing, ), scale_factor = 1.0)
    gen_fn = get_gen_fn(tr)
    ps = get_params(gen_fn)
    ps isa Empty && return
    as, pgs = get_learnable_gradients(ps, get_record(tr), retgrad...; scaler = scale_factor)
    accumulate!(gen_fn.params_grads, pgs)
end

# ------------ Optimization ------------ #

function init_update_state(conf::Gen.FixedStepGradientDescent,
                           gen_fn::JFunction, 
                           param_list::Vector{T}) where T <: Tuple
    Gen.FixedStepGradientDescentBuiltinDSLState(conf.step_size, gen_fn, param_list)
end
function init_update_state(conf::Gen.GradientDescent,
                           gen_fn::JFunction, 
                           param_list::Vector{T}) where T <: Tuple
    Gen.GradientDescentBuiltinDSLState(conf.step_size_init, conf.step_size_beta, gen_fn, param_list, 1)
end

# ------------ Convenience macro ------------ #

function _jaynes(def)

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
                      Any)
        end

        # Matches thunks.
    elseif @capture(def, () -> body__)
        trans = quote
            JFunction($def, 
                      (Tuple{}, ),
                      (false, ),
                      false, 
                      Any)
        end

        # Matches lambdas with formals.
    elseif @capture(def, (args_) -> body__)
        if args isa Symbol
            argtypes = [Any]
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
                      Any)
        end
    else
        error("ParseError (@jaynes): requires a longdef function definition or an anonymous function definition.")
    end

    trans
end

macro jaynes(expr)
    def = _sugar(expr)
    trans = _jaynes(def)
    esc(trans)
end

# ------------ Utilities ------------ #

function display(jfunc::JFunction; show_all = false)
    println(" ___________________________________\n")
    println("             JFunction\n")
    println(" fn : $(jfunc.fn)")
    println(" arg_types : $(jfunc.arg_types)")
    println(" has_argument_grads : $(jfunc.has_argument_grads)")
    println(" accepts_output_grad : $(jfunc.accepts_output_grad)")
    if show_all
        println(" ___________________________________\n")
        display(get_analysis(jfunc))
    else
        println(" ___________________________________\n")
    end
end

function display(vtr::Gen.VectorTrace{A, K, JTrace}; show_values = true, show_types = false) where {A, K}
    addrs = Any[]
    chd = Dict()
    meta = Dict()
    for (k, v) in get_submaps_shallow(get_choices(vtr))
        collect!((k, ), addrs, chd, v, meta)
    end
    println(" ___________________________________\n")
    println("             Address Map\n")
    if show_values
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(chd[a])")
            elseif haskey(chd, a)
                println(" $(a) = $(chd[a])")
            else
                println(" $(a)")
            end
        end
    elseif show_types
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(typeof(chd[a]))")
            elseif haskey(chd, a)
                println(" $(a) = $(typeof(chd[a]))")
            else
                println(" $(a)")
            end
        end
    elseif show_types && show_values
        for a in addrs
            if haskey(meta, a) && haskey(chd, a)
                println(" $(meta[a]) $(a) = $(chd[a]) : $(typeof(chd[a]))")
            elseif haskey(chd, a)
                println(" $(a) = $(chd[a]) : $(typeof(chd[a]))")
            else
                println(" $(a)")
            end
        end
    else
        for a in addrs
            println(" $(a)")
        end
    end
    println(" ___________________________________\n")
end
