# ------------ Factor ------------ #

@inline function (ctx::ScoreContext)(fn::typeof(factor), arg::A) where A <: Number
    ctx.weight += arg
    return arg
end
