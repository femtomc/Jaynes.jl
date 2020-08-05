# ------------ Factor ------------ #

@inline function (ctx::ParameterBackpropagateContext)(fn::typeof(factor), arg::A) where A <: Number
    ctx.weight += arg
    return arg
end

@inline function (ctx::ChoiceBackpropagateContext)(fn::typeof(factor), arg::A) where A <: Number
    ctx.weight += arg
    return arg
end
