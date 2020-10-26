struct Π{N}
    components::SVector{N, Distribution}
    Π(svec::SVector{N, T}) where {N, T} = new{N}(svec)
end
@inline Product(t...) = Π(SVector(t...))
@inline (m::Π)() = SVector([rand(i) for i in m.components])
