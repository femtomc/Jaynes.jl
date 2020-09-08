for s in [:(+), :(*), :(-), :(/), :(>), :(<), :(>=), :(<=)]
    expr = quote 
        @abstract DiffPrimitives Base.$s(v1, v2) = propagate(v1, v2)
        @abstract DiffPrimitives Base.$s(args...) = propagate(args...)
    end
    Base.eval(@__MODULE__, expr)
end
