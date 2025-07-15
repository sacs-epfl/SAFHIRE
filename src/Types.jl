######################################
### Defines basic types to be used ###
######################################
#
const Poly=Vector{Float64}
const Poly_star=Vector{Float64}
const Mask=Vector{Float64}
const Masked=Float64
const LWE=Tuple{Mask,Masked}
const RLWE=Tuple{Poly,Poly}
const RLWE_star=Tuple{Poly_star,Poly_star}
#
const LWE_Key=Vector{Int64}
const RLWE_Key=Vector{Int64}
const FFT_Key=Vector{ComplexF64}
const FFT_Poly=Vector{ComplexF64}
