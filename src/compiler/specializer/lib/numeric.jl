for s in [:(+), :(*), :(-), :(/), :(>), :(<), :(>=), :(<=)]
    expr = quote 
        @abstract DiffInterpreter Base.$s(args...) = propagate(args...)
    end
    Base.eval(@__MODULE__, expr)
end

for s in [sin, exp, cos, log]
    expr = quote 
        @abstract DiffInterpreter $s(args...) = propagate(args...)
    end
    Base.eval(@__MODULE__, expr)
end
