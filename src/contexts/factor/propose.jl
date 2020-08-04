# ------------ Factor ------------ #

@inline function (ctx::ProposeContext)(fn::typeof(factor), arg::A) where A <: Number
    ctx.score += arg
    ctx.weight += arg
    return arg
end
