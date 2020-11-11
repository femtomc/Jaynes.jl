# Used by backpropagation contexts during pullback computation.
merge(tp1::Tuple{}, tp2::Tuple{}) = tp1
merge(tp1::Tuple{Nothing}, tp2::Tuple{Nothing}) where T = tp1
merge(tp1::NTuple{N, Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]
merge(tp1::Array{Float64}, tp2::NTuple{N, Float64}) where N = [tp1[i] + tp2[i] for i in 1 : N]

# Used during specialization for regenerate/update for caching sites which don't need to be re-visited.
@inline function record_cached!(ctx, addr)
    visit!(ctx, addr)
    sub = get_sub(ctx.prev, addr)
    sc = get_score(sub)
    ctx.score += get_score(sub)
    set_sub!(ctx.tr, addr, sub)
    get_value(sub)
end


