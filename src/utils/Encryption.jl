####################################################
### RLWE/RLWE_star Encryption/decryption schemes ###
####################################################
#
#
"""
    encrypt_poly_to_rlwe(μ::Poly,FFT_s::FFT_Key,σ::Float64)

    Encrypts a polynomial μ in the standard basis to RLWE ciphertext (a,b) with key s.
    The ciphertext is expressed in the torus T.
"""
function encrypt_poly_to_rlwe(μ::Poly,FFT_s::FFT_Key,σ::Float64)
    #
    # Encryption of message μ with key s : b=μ + s*(Ω0_star_inv*a_star) + Ω0_star_invinv*e_star 
    # where : 
    #   1. e_star and a_star are in Τ⋆ (picked at random, Gaussian and uniform respectively)
    #   2. s is in R
    #   3. a_star is picked at random in Τ* so that s*a_star ∈ Τ*
    #   4. both a and b are in Τ
    #
    a_star= rand(N)
    FFT_a=FFT_Ω0_star_inv.*(op_fft_M*(Ω_star_to_Ω(a_star)))
    a=real_to_torus.(mod_Φ(real.(op_ifft_M*(FFT_a))))
    #
    e_star=σ*randn(Float64,N)
    e=real.(op_ifft_M*(FFT_Ω0_star_inv.*(op_fft_M*(Ω_star_to_Ω(e_star)))))
    #
    b=real_to_torus.(mod_Φ(μ+real.(op_ifft_M*(FFT_a.*FFT_s))+e))
    #
    return (a,b)
end

"""
    decrypt_rlwe_to_poly(c::RLWE,FFT_s::FFT_Key)

    Decrypts a RLWE ciphertext c=(a,b) to polynomial μ in the standard basis.
"""
function decrypt_rlwe_to_poly(c::RLWE,FFT_s::FFT_Key)
    #
    # Decryption of ciphertext c=(a,b) 
    # where : 
    #   1. μ=b-s*a-e
    #   2. μ, a, b are in T
    #   3. s is in R
    #
    return real_to_torus.(mod_Φ((c[2]-real.(op_ifft_M*(FFT_s.*(op_fft_M*(c[1])))))))
end
#
function encrypt_poly_star_to_rlwe_star(μ_star::Poly_star,FFT_s::FFT_Key,σ::Float64)
    #
    # Encryption of message μ_star with key s : b_star=μ_star + s*a_star + e_star 
    # where : 
    #   1. μ_star, b_star, e_star and a_star are in T* (last two picked at random, Gaussian and uniform respectively)
    #   2. s is in R
    #   3. a_star is picked at random in T_star so that s*a_star ∈ T*
    #
    #   The ouput ciphertext is expressed in T*
    #
    a_star=[rand(N);zeros(M-N)]
    #
    # Multiplies a_star and s, adds to e_star and μ_star
    #
    s_times_a_star=Ω_to_Ω_star(real.(op_ifft_M*(FFT_s.*(op_fft_M*(Ω_star_to_Ω(a_star))))))
    b_star=real_to_torus.(μ_star+s_times_a_star+σ.*[randn(Float64,N);zeros(M-N)]) # Added a_star and b_star to be of lenght M
    #
    return (a_star,b_star) # Expressed in T*
end
#
function decrypt_rlwe_star_to_poly_star(c_star::RLWE_star,FFT_s::FFT_Key)
    #
    # Decryption of ciphertext c_star=(a_star,b_star) 
    # where : 
    #   1. μ_star=b_star-s*a_star-e_star
    #   2. μ_star, a_star, b_star are in Τ*
    #   3. s is in R
    #
    s_times_a=real.(op_ifft_M*(FFT_s.*(op_fft_M*(Ω_star_to_Ω(c_star[1])))))
    return real_to_torus.(c_star[2]-Ω_to_Ω_star(s_times_a))
end
#