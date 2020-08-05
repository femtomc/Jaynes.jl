# ------------ Factor ------------ #

@inline function (ctx::SimulateContext)(fn::typeof(factor), arg)
    ctx.score += arg
    return arg
end
