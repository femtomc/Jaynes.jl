# ------------ Factor ------------ #

@inline function (ctx::ForwardModeContext)(fn::typeof(factor), arg)
    ctx.weight += arg
    return arg
end
