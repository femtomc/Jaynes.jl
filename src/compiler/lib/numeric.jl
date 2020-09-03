for s in [:(+), :(*), :(-), :(/)]
    expr = quote 
        function Base.$s(v::Diffed{K, DK}, q::Diffed{K, DV}) where {K, DK, DV}
            Diffed(v.val + q.val, propagate(DK, DV))
        end
        @abstract DiffPrimitives function Base.$s(v::Diffed{K, DK}, q::Diffed{K, DV}) where {K, DK, DV}
            Diffed{K, propagate(DK, DV)}
        end
    end
    Base.eval(@__MODULE__, expr)
end
