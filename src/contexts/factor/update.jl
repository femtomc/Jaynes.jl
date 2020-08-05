# ------------ Factor ------------ #

@inline function (ctx::UpdateContext)(fn::typeof(factor), arg::A) where A <: Number
    ctx.weight += arg
    ctx.score += arg
    return arg
end
