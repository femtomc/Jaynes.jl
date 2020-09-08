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
get_retval(trace::JTrace) = get_retval(trace.record)
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
end

function JFunction(arg_types::NTuple{N, Type},
                        func::Function,
                        has_argument_grads::NTuple{N, Bool},
                        accepts_output_grad::Bool,
                        ::Type{R}) where {N, R}
    JFunction{N, R}(func, DynamicMap{Value}(), DynamicMap{Value}(), arg_types, has_argument_grads, accepts_output_grad)
end

function (jfn::JFunction)(args...)
    jfn.fn(args...)
end

has_argument_grads(jfn::JFunction) = jfn.has_argument_grads
get_params(jfn::JFunction) = jfn.params
init_param!(jfn, addr, val) = set_sub!(jfn.params, addr, Value(val))
init_param!(jfn, v::Vector{Pair{T, K}}) where {T <: Tuple, K} = begin
    for (addr, val) in v
        init_param!(jfn, addr, val)
    end
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

function assess(jfn::JFunction, args::Tuple, choices::JChoiceMap)
    ret, w = score(unwrap(choices), 
                   get_params(jfn), 
                   jfn.fn, 
                   args...)
    w, ret
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
                               args, 
                               arg_diffs)
    JTrace(get_gen_fn(trace), cl, false), w, rd, JChoiceMap(d)
end
@inline update(trace::JTrace, args::Tuple, arg_diffs::Tuple, constraints::DynamicChoiceMap) = update(trace, args, arg_diffs, JChoiceMap(convert(DynamicMap{Value}, constraints)))

function regenerate(trace::JTrace, args::Tuple, arg_diffs::Tuple, selection::JSelection)
    ret, cl, w, rd, d = regenerate(unwrap(selection), 
                                   get_params(get_gen_fn(trace)), 
                                   get_record(trace), 
                                   args, 
                                   arg_diffs)
    JTrace(get_gen_fn(trace), cl, false), w, rd, JChoiceMap(d)
end

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

function accumulate_param_gradients!(tr::JTrace, retgrad = nothing, scale_factor = 1.0)
    gen_fn = get_gen_fn(tr)
    ps = get_params(gen_fn)
    as, pgs = get_learnable_gradients(ps, get_record(tr), retgrad...; scaler = scale_factor)
    accumulate!(gen_fn.params_grads, pgs)
end

# ------------ Convenience macro ------------ #

macro jaynes(expr)
    def = _sugar(expr)
    if @capture(def, function decl_(args__) body__ end)
        trans = quote 
            $def
            JFunction((), $decl, (), true, Any)
        end
    else
        trans = quote
            JFunction((), $def, (), true, Any)
        end
    end
    esc(trans)
end
