"""
    compute_RLWE_keyswitching_key(FFT_s::FFT_Key,s_prime::Vector{Int64},σ_ksw::Float64,D::Int64,l::Int64)

    Compute the keyswitching secret to perform the key switching from s_prime to s

    Args:
        FFT_s::FFT_Key: The FFT key for the secret key s.
        s_prime::Vector{Int64}: The secret key s_prime.
        σ_ksw::Float64: The standard deviation for the noise.
        D::Int64: The scaling factor.
        l::Int64: The number of levels.

    Returns:
        ksk_s_prime_to_s::Array{RLWE}: The keyswitching secret to perform the key switching from s_prime to s.
"""
function compute_rlwe_keyswitching_key(FFT_s::FFT_Key,s_prime::Vector{Int64},σ_ksw::Float64,D::Int64,l::Int64)
    #
    # Compute the keyswitching secret to perform the key switching from s_prime to s
    #
    ksk_s_prime_to_s = Array{RLWE}(undef,l)
    for j = 1:l
        ksk_s_prime_to_s[j] = encrypt_poly_to_rlwe(s_prime/D^j,FFT_s,σ_ksw)
    end
    #
    return ksk_s_prime_to_s
end

function rlwe_key_switch(
                        C_prime::RLWE,
                        FFT_ksk_s_prime_to_s::Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}},
                        D::Int64,l::Int64
                        )
    #
    # Key switch a LWE cyphertext encrypter with key s_prime to key s
    #
    C=RLWE
    n_prime=length(FFT_ksk_s_prime_to_s[1][1])
    a = zeros(ComplexF64, n_prime)
    b = zeros(ComplexF64, n_prime)
    #
    aux_a = zeros(Float64, n_prime, l)
    temp = zeros(l)
    for j=1:n_prime
        temp .= decompose_torus(C_prime[1][j], D, l)
        for i = 1:l
            aux_a[j,i] = convert(Float64, temp[i])
        end
    end

    for i=1:l
        FFT_aux = (op_fft_M*(aux_a[:,i]))
        a .-= FFT_aux .* FFT_ksk_s_prime_to_s[i][1]
        b .-= FFT_aux .* FFT_ksk_s_prime_to_s[i][2]
        FFT_aux = nothing
    end

    # Manually clear references
    aux_a = nothing
    
    C=(real.(op_ifft_M*a),real.(op_ifft_M*b)+C_prime[2])
    
    a = nothing
    b = nothing

    return C
end

function rlwe_key_switch_accumulate!(
                        a_sum::Vector{Float64}, b_sum::Vector{Float64},
                        a_in::Vector{Float64},b_in::Vector{Float64}, 
                        FFT_ksk_s_prime_to_s::Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}},
                        D::Int64,l::Int64
                        )
    #
    # Key switch a LWE cyphertext encrypter with key s_prime to key s
    #
    n_prime=length(FFT_ksk_s_prime_to_s[1][1])
    a = zeros(ComplexF64, n_prime)
    b = zeros(ComplexF64, n_prime)
    #
    aux_a = zeros(Float64, n_prime, l)
    temp = zeros(l)
    @inbounds for j=1:n_prime
        temp .= decompose_torus(a_in[j], D, l)
        @inbounds for i = 1:l
            aux_a[j,i] = convert(Float64, temp[i])
        end
    end

    @inbounds for i=1:l
        FFT_aux = (op_fft_M*(aux_a[:,i]))
        a .-= FFT_aux .* FFT_ksk_s_prime_to_s[i][1]
        b .-= FFT_aux .* FFT_ksk_s_prime_to_s[i][2]
    end
    
    b_sum .+= b_in
    a_sum .+= real.(op_ifft_M*a)
    b_sum .+= real.(op_ifft_M*b)
end