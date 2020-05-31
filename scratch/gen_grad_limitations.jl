module GenGradLimitations

using Gen

@gen function disc_model(x::Float64)
    @param a::Float64
    z = a + 10.0
    @param b::Float64
    @trace(normal(z + b, 1.), :y)
end

init_param!(disc_model, :a, 5.0)
init_param!(disc_model, :b, 5.0)

end # module
