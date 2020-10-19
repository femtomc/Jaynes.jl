for s in [:(+), :(*), :(-), :(/), :(>), :(<), :(>=), :(<=)]
    expr = quote 
        @abstract DiffPrimitives Base.$s(args...) = propagate(args...)
    end
    Base.eval(@__MODULE__, expr)
end

for s in [sin, exp, cos, log]
    expr = quote 
        @abstract DiffPrimitives $s(args...) = propagate(args...)
    end
    Base.eval(@__MODULE__, expr)
end
