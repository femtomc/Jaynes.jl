# ------------ Factor ------------ #

@inline function (ctx::GenerateContext)(fn::typeof(factor), arg)
    ctx.score += arg
    ctx.weight += arg
    return arg
end
