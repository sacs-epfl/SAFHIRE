const array_FFT_ksk_s_d_to_s = Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}}(undef, M)
indices = compute_automorphism_indices(g)
for d in indices
    array_FFT_ksk_s_d_to_s[d] = Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}(undef, l)
    for i=1:l
        array_FFT_ksk_s_d_to_s[d][i] = (CuArray(op_fft_M*array_ksk_s_d_to_s[d][i][1]), CuArray(op_fft_M*array_ksk_s_d_to_s[d][i][2]))
    end
end

const batch_size = 40
plan_ref = Ref{CUDA.CUFFT.cufftHandle}()
CUDA.CUFFT.cufftPlan1d(plan_ref, M, CUDA.CUFFT.CUFFT_Z2Z, batch_size)

function decompose_torus_batch_gpu_fast!(x_gpu::CuArray{ComplexF64},μ_batch::CuArray{ComplexF64}, D::Int64, l::Int64)
    M, batch_size = size(μ_batch)

    # Project μ_batch onto the torus
    μ_batch = μ_batch .- round.(real.(μ_batch)) .+ im .* imag.(μ_batch)

    @cuda threads=1024 blocks=ceil(Int, M * batch_size / 256) kernel_decompose!(μ_batch, x_gpu, D, l, M, batch_size)
    
    return
end

function kernel_decompose!(μ_batch, x_gpu, D, l, M, batch_size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = M * batch_size

    if idx > total
        return
    end

    # Flattened indexing into μ_batch
    i = fld(idx - 1, batch_size) + 1  # row (1 to M)
    j = (idx - 1) % batch_size + 1    # col (1 to batch_size)

    xtilde = μ_batch[i, j]

    for k in 1:l
        rounded = ComplexF64(round(D * real(xtilde)), 0.0)
        x_gpu[i, j, k] = rounded
        xtilde = D * xtilde - rounded
    end
end
function batch_multi_pol_sub_fused!(
    aux::CuArray{ComplexF64},
    FFT_Q1::CuArray{ComplexF64},
    FFT_Q2::CuArray{ComplexF64},
    A_batch::CuArray{ComplexF64},
    B_batch::CuArray{ComplexF64}
)
    # In-place FFT
    Mf = float(M)
    CUDA.CUFFT.cufftExecZ2Z(plan_ref[], aux, aux, CUDA.CUFFT.CUFFT_FORWARD)

    # Perform pointwise multiplications
    FFT_Q1_reshape = reshape(FFT_Q1, :, 1)
    FFT_Q2_reshape = reshape(FFT_Q2, :, 1)
    @. A_batch -= (aux * FFT_Q1_reshape) / Mf
    @. B_batch -= (aux * FFT_Q2_reshape) / Mf

    return nothing
end
#
"""
    function gpu_rlwe_key_switch(A_batch_prime::CuArray{ComplexF64}, B_batch_prime::CuArray{ComplexF64},
                               A_batch::CuArray{ComplexF64}, B_batch::CuArray{ComplexF64},
                               FFT_ksk_s_prime_to_s::Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}},
                               D::Int64, l::Int64)

    Key switch a batch of RLWE cyphertexts encrypted with key s_prime to key s

    Args:
        A_batch_prime::CuArray{ComplexF64}: The first part of the RLWE cyphertext to be key switched.
        B_batch_prime::CuArray{ComplexF64}: The second part of the RLWE cyphertext to be key switched.
        A_batch::CuArray{ComplexF64}: The first part of the output RLWE cyphertext.
        B_batch::CuArray{ComplexF64}: The second part of the output RLWE cyphertext.
        FFT_ksk_s_prime_to_s::Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}: The FFT of the keyswitching secret to perform the key switching from s_prime to s.
        D::Int64: The scaling factor.
        l::Int64: The number of levels.
        aux_a_gpu: A buffer for the leveled decomposition
    Returns:
        nothing: The function modifies the input A_batch and B_batch in place.
"""
function gpu_batch_rlwe_key_switch(A_batch_prime::CuArray{ComplexF64}, B_batch_prime::CuArray{ComplexF64},
                               A_batch::CuArray{ComplexF64}, B_batch::CuArray{ComplexF64},
                               FFT_ksk_s_prime_to_s::Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}},
                               D::Int64, l::Int64, aux_a_gpu::CuArray{ComplexF64})
    #
    # Key switch a RLWE cyphertext encrypter with key s_prime to key s
    #
    n_prime=length(FFT_ksk_s_prime_to_s[1][1])

    # === Torus Decomposition ==
    decompose_torus_batch_gpu_fast!(aux_a_gpu,A_batch_prime, D, l)

    for i=1:l
        batch_multi_pol_sub_fused!(
            aux_a_gpu[:, :, i],
            FFT_ksk_s_prime_to_s[i][1],
            FFT_ksk_s_prime_to_s[i][2],
            A_batch,
            B_batch
        )
    end

    # Inverse FFT on both result buffers
    CUDA.CUFFT.cufftExecZ2Z(plan_ref[], A_batch, A_batch, CUDA.CUFFT.CUFFT_INVERSE)
    CUDA.CUFFT.cufftExecZ2Z(plan_ref[], B_batch, B_batch, CUDA.CUFFT.CUFFT_INVERSE)


    # Add the second part of ciphertext (B)
    B_batch .+= B_batch_prime

    return nothing
end