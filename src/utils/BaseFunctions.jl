#################################
### Some elementary functions ###
#################################
#
function roundup(x::Float64)
    #
    # Rounds to nearest integer 
    #
    convert(Int64,round(x,RoundNearestTiesUp))
end
#
function rounddown(x::Float64)
    #
    # Rounds to nearest integer 
    #
    convert(Int64,-round(-x,RoundNearestTiesUp))
end
#
"""
    function Real_to_Torus(μ::Float64)
    
    Converts a real to its main representative in the torus [-1/2,1/2[

    Args:
        μ::Float64: The real number to be converted.
    Returns:
        Float64: The representative in the torus.
"""
function real_to_torus(μ::Float64)
    #
    # Convert a real to its main representative in the torus [-1/2,1/2[
    #
    μ=mod(μ,1)
    return(μ-(μ>=1/2))
end
function real_to_torus(A::Array{Float64, 2})
    M,B = size(A)
    A_out = Array{Float64}(undef, M, B)
    LoopVectorization.@tturbo for m in 1:M, b in 1:B
        μ=mod(A[m,b],1)
        A_out[m,b] = μ-(μ>=1/2)
    end
    return A_out
end
function project_to_discrete_torus(μ::Float64,p::Int64)
    #
    # Projects μ on the discrete torus Τ_p
    # 
    return real_to_torus(convert(Int64,rounddown(μ*p))/p)
end
#
"""
    function Decompose_Torus(μ::Float64,D::Int64,ℓ::Int64)
    
    Decomposes μ as ∑_{i=1}^ℓ x[i] D^{-i}

    Args:
        μ::Float64: The real number to be decomposed.
        D::Int64: The base of the decomposition.
        ℓ::Int64: The number of terms in the decomposition.
    Returns:
        Vector{Int64}: The coefficients of the decomposition.
"""
function decompose_torus(μ::Float64,D::Int64,ℓ::Int64)
    #
    # Decomposes μ as ∑_{i=1}^ℓ x[i] D^{-i}
    #
    μ=real_to_torus(μ)
    x=Vector{Int64}(undef,ℓ)
    xtilde=μ
    for i in 1:ℓ
        x[i]=convert(Int64,roundup(D*xtilde))
        xtilde=D*xtilde-x[i]
    end
    return x    
end

"""
    function Multi_Pol(P::Poly,Q::Poly)

    Multiplies polynomials P and Q mod X^M-1 using FFTW.

    Args:
        P::Poly: The first polynomial.
        Q::Poly: The second polynomial.
    Returns:
        Poly: The product of P and Q mod X^M-1.
"""
function multi_pol(P::Poly,Q::Poly)
    #
    # Multiplies polynomials P and Q mod X^M-1 
    #
    return real.(op_ifft_M*((op_fft_M*(P)).*(op_fft_M*(Q))))
end
function multi_pol(FFT_P::Vector{ComplexF64},Q::Poly)
    #
    # Multiplies polynomials P and Q mod X^M-1 
    #
    return real.(op_ifft_M*(FFT_P.*(op_fft_M*(Q))))
end
function multi_pol(P::Poly, FFT_Q::Vector{ComplexF64})
    return multi_pol(FFT_Q, P)
end
function multi_pol(FFT_P::Vector{ComplexF64},FFT_Q::Vector{ComplexF64})
    #
    # Multiplies polynomials P and Q mod X^M-1 
    #
    return real.(op_ifft_M*(FFT_P.*FFT_Q))
end
#
"""
    function Mod_Φ(Q::Poly)
    
    Computes Q mod Φ_M
    
    Args:
        Q::Poly: The polynomial to be reduced.
    Returns:
        Q_mod::Poly: The reduced polynomial.
"""
function mod_Φ(Q::Poly)
    #
    # Computes Q mod Φ_M
    #
    return Q-kron(one_t,Q[N+1:M])
end
#
const Mod_Table=Vector{Int64}(undef,N)
for i in 1:N 
    Mod_Table[i]=mod(i-1,M_over_t)+1
end

"""
    function Ω_to_Ω_star(P::Poly)
    
    Converts a polynomial in the standard basis to the star basis.
    The polynomial is represented as a vector of coefficients in the standard basis.

    Args:
        P::Poly: The polynomial in the standard basis.
    Returns:
        P_star::Poly_star: The polynomial in the star basis.
"""
function Ω_to_Ω_star(P::Poly)
    #
    # Determines the coefficients P_star[i] of P(X)=∑ P_star[i+1] Ω_star[i](X) from P(X)=∑ P[i+1] Ω[i](X)
    #
    P=Mod_Φ(P)
    P_star=[M*P[1:N]; zeros(M-N)]
    ######
    ΣP=zeros(M_over_t)
    for i in 1:M_over_t
        for k in 0:t-2
            ΣP[i]+=P[i+k*M_over_t]
        end
        ΣP[i]*=M_over_t
    end
    for i ∈ 1:N
        P_star[i]-=ΣP[Mod_Table[i]]
    end
    return P_star
end

"""
    function Ω_star_to_Ω(P_star::Poly_star)
    
    Converts a polynomial in the star basis to the standard basis.
    The polynomial is represented as a vector of coefficients in the star basis.

    Args:
        P_star::Poly_star: The polynomial in the star basis.
    Returns:
        P::Poly: The polynomial in the standard basis.
"""
function Ω_star_to_Ω(P_star::Poly_star)
    #
    # Determines the coefficients P[i] of P(X)=∑ P[i] Ω[i](X) from P(X)=∑ P_star[i] Ω_star[i](X) 
    #
    P=zeros(M)
    ΣP_star=zeros(M_over_t)
    for i in 1:M_over_t
        for k in 0:t-2
            ΣP_star[i]+=P_star[i+k*M_over_t]
        end
    end
    for i ∈ 1:N
        P[i]=P_star[i]+ΣP_star[Mod_Table[i]]
    end
    return F*P
end
#
function Ω_star_to_Ω_int(P_star::Poly_star)
    #
    # Determines the coefficients P[i] of P(X)=∑ P[i] Ω[i](X) from P(X)=∑ P_star[i] Ω_star[i](X) 
    #
    P=zeros(M)
    ΣP_star=zeros(M_over_t)
    for i in 1:M_over_t
        for k in 0:t-2
            ΣP_star[i]+=P_star[i+k*M_over_t]
        end
    end
    for i ∈ 1:N
        P[i]=P_star[i]+ΣP_star[Mod_Table[i]]
    end
    return M_inv*P
end

function compute_automorphism_indices(g::Int64)
    indices = []
    π = t-1
    aux = 2
    while π > 1
        while !(mod(π, aux)==0)
            aux = nextprime(aux+1)
        end
        π = π ÷ aux
        for k=1:(aux-1)
            d = powermod(g, k*π , M)
            push!(indices, d)
        end
    end
    for j=1:(α-1)
        for k=1:(t-1)
            d = k*t^j + 1
            push!(indices, d)
        end
    end
    return indices
end

function get_conv_output_size(input_shape::NTuple{4, Int64}, kernel_shape::NTuple{4, Int64}, stride::Int64, padding::Int64)
    # Calculate the output size of a convolution operation
    H, W, C_in, B = input_shape
    KH, KW, C_in, C_out = kernel_shape

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    H_out = (H_pad - KH) ÷ stride + 1
    W_out = (W_pad - KW) ÷ stride + 1

    in_dim = B * H * W * C_in
    out_dim = B * H_out * W_out * C_out

    return (in_dim, out_dim)
end

# A version of fill! zeros for multithreaded performances
function multi_threaded_fill_zeros!(A::AbstractMatrix{Float64},
    i_block_size::Int = 4)
    M, B = size(A)
    Threads.@threads for i_block in 1:i_block_size:B
        i_end = min(i_block + i_block_size - 1, B)
        for b in i_block:i_end
            @inbounds @simd for m=1:M
                A[m,b] = 0.0
            end
        end
    end
    return
end
function simd_blocked_transpose!(B::AbstractMatrix{T}, A::AbstractMatrix{T}, blocksize::Int=64) where T
    m, n = size(A)
    @assert size(B) == (n, m)
    
    Threads.@threads for ii in 1:blocksize:m
        for jj in 1:blocksize:n
            i_max = min(ii + blocksize - 1, m)
            j_max = min(jj + blocksize - 1, n)
            LoopVectorization.@tturbo for i in ii:i_max, j in jj:j_max
                    B[j, i] = A[i, j]
            end
        end
    end
end