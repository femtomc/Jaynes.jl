# ------------ GFI interface implementations for Jaynes ------------ #

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
get_sub(choices::JChoiceMap, addr) = get_sub(unwrap(choices), addr)
projection(choices::JChoiceMap, sel) = projection(unwrap(choices), sel)

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

@inline collect!(par::T, addrs::Vector, chd::Dict, chm::JChoiceMap, meta) where T <: Tuple = collect!(par, addrs, chd, unwrap(chm), meta)

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

@inline convert(::Type{DynamicMap{Value}}, d::JChoiceMap) = convert(DynamicMap{Value}, unwrap(d))

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

mutable struct JTrace{A, R} <: Trace
    chm::JChoiceMap
    score::Float64
    jfn::GenerativeFunction
    args::A
    ret::R
    isempty::Bool
end

@inline get_record(trace::JTrace) = trace.chm
@inline set_retval!(trace::JTrace, retval) = (trace.retval = retval)
@inline has_choice(trace::JTrace, addr) = haskey(trace.record, addr) && is_choice(get_sub(trace.record, addr))
@inline has_value(trace::JTrace, addr) = has_value(get_record(trace), addr)
@inline get_value(trace::JTrace, addr) = getindex(get_record(trace), addr)
@inline filter(fn::Function, tr::JTrace) = JChoiceMap(filter(fn, unwrap(get_record(tr))))

function get_choice(trace::JTrace, addr)
    ch = get_sub(trace.record, addr)
    !is_choice(ch) && throw(KeyError(addr))
    ch
end

get_choices(trace::JTrace) = get_record(trace)

Base.display(jtr::JTrace) = Base.display(get_record(jtr))

# Trace GFI methods.
get_args(trace::JTrace) = trace.args
get_retval(trace::JTrace) = trace.ret
get_score(trace::JTrace) = trace.score
get_gen_fn(trace::JTrace) = trace.jfn

# ------------ Generative function ------------ #

struct JFunction{C <: CompilationOptions, N, R, T} <: TypedGenerativeFunction{N, R, JTrace, T}
    fn::Function
    params::DynamicMap{Value}
    params_grad::DynamicMap{Value}
    arg_types::NTuple{N, Type}
    has_argument_grads::NTuple{N, Bool}
    accepts_output_grad::Bool
    ir::IR
    reachability::FlowAnalysis
    trace_type::T
end

function JFunction(opt::J,
                   func::Function,
                   arg_types::NTuple{N, Type},
                   has_argument_grads::NTuple{N, Bool},
                   accepts_output_grad::Bool,
                   ::Type{R}) where {J <: CompilationOptions, N, R}
    (tt, ir) = instantiation_pipeline(func, arg_types, R, opt)
    return JFunction{J, N, R, typeof(tt)}(func, 
                                          DynamicMap{Value}(), 
                                          DynamicMap{Value}(), 
                                          arg_types, 
                                          has_argument_grads, 
                                          accepts_output_grad,
                                          ir,
                                          flow_analysis(ir),
                                          tt)
end

@inline (jfn::JFunction)(args...) = jfn.fn(args...)
@inline has_argument_grads(jfn::JFunction) = jfn.has_argument_grads
@inline get_params(jfn::JFunction) = jfn.params
@inline get_param(jfn::JFunction, name) = getindex(jfn.params, name)
@inline set_param!(jfn::JFunction, name, v) = set_sub!(jfn.params, name, Value(v))
@inline get_params_grad(jfn::JFunction) = jfn.params_grad
@inline get_param_grad(jfn::JFunction, name) = getindex(jfn.params_grad, name)
@inline zero_param_grad!(jfn::JFunction, name) = set_sub!(jfn.params_grad, name, Value(zero(jfn.params[name])))
@inline set_param_grad!(jfn::JFunction, name::Symbol, v) = set_sub!(jfn.params_grad, name, Value(v))

init_param!(jfn, addr, val) = set_sub!(jfn.params, addr, Value(val))
init_param!(jfn, v::Vector{Pair{T, K}}) where {T <: Tuple, K} = begin
    for (addr, val) in v
        init_param!(jfn, addr, val)
    end
end

# Jaynes extensions.
@inline get_fn(jfn::JFunction) = jfn.fn
@inline get_analysis(jfn::JFunction) = jfn.reachability
@inline get_ir(jfn::JFunction) = jfn.ir
@inline get_trace_type(jfn::JFunction{C, N, R, T}) where {C, N, R, T} = T
@inline get_opt_type(::JFunction{C}) where C = C

# Typed JFunction instances have a defined notion of AC (absolute continuity).
@inline Base.:(<<)(jfn1::JFunction{N1, R1, T1}, jfn2::JFunction{N2, R2, T2}) where {N1, N2, R1, R2, T1, T2} = T1 << T2

# ------------ Model GFI interface ------------ #

function simulate(jfn::JFunction{C}, args::Tuple) where C
    ret, cl = simulate(C(),
                       get_params(jfn), 
                       jfn.fn, 
                       args...)
    JTrace(get_trace(cl) |> JChoiceMap, cl.score, jfn, args, ret, false)
end

function generate(jfn::JFunction{C}, args::Tuple, chm::JChoiceMap) where C
    ret, cl, w = generate(C(),
                          unwrap(chm), 
                          get_params(jfn), 
                          jfn.fn, 
                          args...)
    JTrace(get_trace(cl) |> JChoiceMap, cl.score, jfn, args, ret, false), w
end
@inline generate(jfn::JFunction, args::Tuple, choices::DynamicChoiceMap) = generate(jfn, args, JChoiceMap(convert(DynamicMap{Value}, choices)))
@inline generate(jfn::JFunction, args::Tuple, choices::DynamicMap) = generate(jfn, args, JChoiceMap(choices))

function assess(jfn::JFunction{C}, args::Tuple, choices::JChoiceMap) where C
    ret, w = assess(C(),
                    unwrap(choices), 
                    get_params(jfn), 
                    jfn.fn, 
                    args...)
    w, ret
end
@inline assess(jfn::JFunction, args::Tuple, choices::DynamicChoiceMap) = assess(jfn, args, JChoiceMap(convert(DynamicMap{Value}, choices)))

# TODO: must accept any inheritor of Selection.
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
    ret, cl, w, rd, d = update(get_opt_type(get_gen_fn(trace))(),
                               unwrap(constraints), 
                               get_params(get_gen_fn(trace)), 
                               DynamicCallSite(unwrap(get_record(trace)),
                                               get_score(trace), 
                                               get_gen_fn(trace).fn, 
                                               get_args(trace), 
                                               get_retval(trace)),
                               map(zip(args, arg_diffs)) do (a, d)
                                   Diffed(a, d)
                               end...)
    JTrace(get_trace(cl) |> JChoiceMap, cl.score, get_gen_fn(trace), args, ret, false), w, rd, JChoiceMap(d)
end
@inline update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::DynamicChoiceMap) = update(trace, args, arg_diffs, JChoiceMap(convert(DynamicMap{Value}, constraints)))
@inline update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::StaticMap) = update(trace, args, arg_diffs, JChoiceMap(constraints))

function regenerate(trace::JTrace, args::Tuple, arg_diffs::Tuple, selection::JSelection)
    ret, cl, w, rd, d = regenerate(get_opt_type(get_gen_fn(trace))(),
                                   unwrap(selection), 
                                   get_params(get_gen_fn(trace)), 
                                   DynamicCallSite(unwrap(get_record(trace)),
                                                   get_score(trace), 
                                                   get_gen_fn(trace).fn, 
                                                   get_args(trace), 
                                                   get_retval(trace)),
                                   map(zip(args, arg_diffs)) do (a, d)
                                       Diffed(a, d)
                                   end...)
    JTrace(get_trace(cl) |> JChoiceMap, cl.score, get_gen_fn(trace), args, ret, false), w, rd, JChoiceMap(d)
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

function get_learnable_gradients(ps::P, tr::JTrace, ret_grad...; scaler::Float64 = 1.0) where P <: AddressMap
    param_grads = Gradients()
    cl = DynamicCallSite(get_record(tr), get_score(tr), get_gen_fn(tr).fn, get_args(tr), get_retval(tr)) 
    arg_grads = accumulate_learnable_gradients!(target(), ps, param_grads, cl, ret_grad...; scaler = scaler)
    return arg_grads, param_grads
end

function accumulate_param_gradients!(tr::JTrace, retgrad = nothing, scale_factor = 1.0)
    gen_fn = get_gen_fn(tr)
    ps = get_params(gen_fn)
    ps isa Empty && return
    as, pgs = get_learnable_gradients(ps, tr, retgrad; scaler = scale_factor)
    accumulate!(gen_fn.params_grad, pgs)
end

# ------------ Optimization ------------ #

# Fixed step size gradient descent.
mutable struct FixedStepGradientDescentJState
    step_size::Float64
    jfn::JFunction
    param_list::Vector
end

function init_update_state(conf::Gen.FixedStepGradientDescent,
                           jfn::JFunction, 
                           param_list::Vector{T}) where T <: Tuple
    FixedStepGradientDescentJState(conf.step_size, jfn, param_list)
end

function apply_update!(state::FixedStepGradientDescentJState)
    for param_name in state.param_list
        value = get_param(state.jfn, param_name)
        grad = get_param_grad(state.jfn, param_name)
        set_param!(state.jfn, param_name, value + grad * state.step_size)
        zero_param_grad!(state.jfn, param_name)
    end
end

# Gradient descent.
mutable struct GradientDescentJState
    step_size_init::Float64
    step_size_beta::Float64
    jfn::JFunction
    param_list::Vector
    t::Int
end

function init_update_state(conf::Gen.GradientDescent,
                           jfn::JFunction, 
                           param_list::Vector{T}) where T <: Tuple
    GradientDescentJState(conf.step_size_init, conf.step_size_beta, jfn, param_list, 1)
end

function apply_update!(state::GradientDescentJState)
    step_size = state.step_size_init * (state.step_size_beta + 1) / (state.step_size_beta + state.t)
    for param_name in state.param_list
        value = get_param(state.jfn, param_name)
        grad = get_param_grad(state.jfn, param_name)
        set_param!(state.jfn, param_name, value + grad * step_size)
        zero_param_grad!(state.jfn, param_name)
    end
    state.t += 1
end

# ------------ Utilities ------------ #

function display(jfn::JFunction{C, N, R, T}; show_all = false) where {C, N, R, T}
    println(" ___________________________________\n")
    println("             JFunction\n")
    println(" fn : $(jfn.fn)")
    println(" arg_types : $(jfn.arg_types)")
    println(" ret_type : $(R)")
    println(" trace_type: $(T)")
    println(" has_argument_grads : $(jfn.has_argument_grads)")
    println(" accepts_output_grad : $(jfn.accepts_output_grad)")
    println("\n compilation options: $C")
    if show_all
        println(" ___________________________________\n")
        display(get_analysis(jfn))
    else
        println(" ___________________________________\n")
    end
end
