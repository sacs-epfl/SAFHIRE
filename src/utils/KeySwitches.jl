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


# If several threads are used prepare for big batched convolution
if Threads.nthreads() > 1
    const cpu_batch_size = 64
else
    const cpu_batch_size = 1
end
const batch_fft_plans = Vector{Any}(undef, cpu_batch_size)
const batch_ifft_plans = similar(batch_fft_plans)
FFTW.set_num_threads(Threads.nthreads())   
for b in 1:cpu_batch_size
    X = rand(ComplexF64, M, b, l)
    batch_fft_plans[b] = plan_fft!(X, 1; flags=FFTW.MEASURE)
    X = rand(ComplexF64, M, b)
    batch_ifft_plans[b] = plan_ifft!(X, 1; flags=FFTW.MEASURE)
end

function batch_rlwe_key_switch_accumulate!(
    a_sum::Array{Float64}, b_sum::Array{Float64},
    a_in::Array{Float64},b_in::Array{Float64}, 
    FFT_ksk_s_prime_to_s::Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}},
    D::Int64,l::Int64
)
    #
    # Key switch a LWE cyphertext encrypter with key s_prime to key s
    #
    n_prime=length(FFT_ksk_s_prime_to_s[1][1])
    b_size = size(a_sum, 2)

    a_rlwe = zeros(ComplexF64, n_prime,b_size)
    b_rlwe = zeros(ComplexF64, n_prime,b_size)
    #
    aux_a = zeros(ComplexF64, n_prime, b_size, l)

    @inbounds Threads.@threads for idx=1:n_prime*b_size
        j = (idx - 1) % M + 1          # Fastest-varying dimension
        b = (idx - 1) ÷ M + 1
        xtilde = a_in[j, b]
        @inbounds for k in 1:l
            rounded = round(D * xtilde)
            aux_a[j, b, k] = ComplexF64(rounded, 0.0)
            xtilde = D * xtilde - rounded
        end
    end
    #@time "fft" begin
        
    batch_fft_plans[b_size]*(aux_a)
    @inbounds for i=1:l
        @views a_rlwe .-= aux_a[:,:,i] .* reshape(FFT_ksk_s_prime_to_s[i][1],:)
        @views b_rlwe .-= aux_a[:,:,i] .* reshape(FFT_ksk_s_prime_to_s[i][2],:)
    end

    
    b_sum .+= b_in
    batch_ifft_plans[b_size]*a_rlwe
    batch_ifft_plans[b_size]*b_rlwe

    a_sum .+= real.(a_rlwe)
    b_sum .+= real.(b_rlwe)

end

function update_rlwe_threaded!(
    a_rlwe::Matrix{ComplexF64}, b_rlwe::Matrix{ComplexF64},
    aux_a::Array{ComplexF64, 3}, FFT_ksk::AbstractArray{ComplexF64}
)
    N, M, l = size(aux_a)  # assuming aux_a is N×M×l
    @assert size(FFT_ksk) == (N, l, 2, 1) || size(FFT_ksk) == (N, l, 2)

    for i = 1:l
    Threads.@threads for col = 1:M
        for row = 1:N
            a_val = aux_a[row, col, i]
            k1 = FFT_ksk[row, i, 1]
            k2 = FFT_ksk[row, i, 2]
            a_rlwe[row, col] -= a_val * k1
            b_rlwe[row, col] -= a_val * k2
        end
    end
    end
end
function batch_rlwe_key_switch_accumulate!(
    a_sum::Array{Float64}, b_sum::Array{Float64},
    a_in::Array{Float64},b_in::Array{Float64}, 
    FFT_ksk_s_prime_to_s::AbstractArray{ComplexF64},
    D::Int64,l::Int64,
    a_rlwe::Array{ComplexF64},
    b_rlwe::Array{ComplexF64},
    aux_a::Array{ComplexF64}
)
    #
    # Key switch a LWE cyphertext encrypter with key s_prime to key s
    #
    b_size = size(a_sum, 2)

    #fill!(a_rlwe, 0.0)
    #fill!(b_rlwe, 0.0)

    # Vectorized and in-place computation for better performance
    @inbounds Threads.@threads for b = 1:b_size
        for j = 1:M
            xtilde = a_in[j, b]
            for k = 1:l
                rounded = round(D * xtilde)
                aux_a[j, b, k] = ComplexF64(rounded, 0.0)
                xtilde = D * xtilde - rounded
            end
        end
    end
    
    batch_fft_plans[b_size]*(aux_a)
    #=@inbounds for i=1:l
        @views a_rlwe .-= aux_a[:,:,i] .* reshape(FFT_ksk_s_prime_to_s[:,i,1],:)
        @views b_rlwe .-= aux_a[:,:,i] .* reshape(FFT_ksk_s_prime_to_s[:,i,2],:)
    end =#
    update_rlwe_threaded!(a_rlwe, b_rlwe, aux_a, FFT_ksk_s_prime_to_s)


    b_sum .+= b_in
    #batch_ifft_plans[b_size]*a_rlwe
    #batch_ifft_plans[b_size]*b_rlwe

    #a_sum .+= real.(a_rlwe)
    #b_sum .+= real.(b_rlwe)
end