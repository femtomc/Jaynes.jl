get_variate_type(d::Type{<: Distribution{T, K}}) where {T, K} = T
get_support_type(d::Type{<: Distribution{T, K}}) where {T, K} = K <: Distributions.Continuous ? Float64 : Int

@abstract DiffPrimitives (::D)(args...) where D <: Distributions.Distribution = AType{<: Distribution{get_variate_type(D), get_support_type(D)}}

@abstract DiffPrimitives rand(::AType{D}) where D <: Distribution = get_support_type(D)
@abstract DiffPrimitives rand(::Symbol, ::AType{D}) where D <: Distribution = get_support_type(D)
@abstract DiffPrimitives rand(::Random._GLOBAL_RNG, ::Distribution{T, K}) where {T, K} = K <: Distributions.Continuous ? Float64 : Int
@abstract DiffPrimitives rand(::Symbol, call::Function, args...) = propagate(args...)

@abstract DiffPrimitives (::Ds)(args::D...) where {Ds <: Distribution, D <:Diffed} = AType{ <: Diff{Ds, UndefinedChange}}
