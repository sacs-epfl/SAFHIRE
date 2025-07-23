function gpu_batch_partial_trace!(A_batch::CuArray{Float64}, B_batch::CuArray{Float64}, j::Int64,
                              array_FFT_ksk_s_d_to_s::Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}},
                              D::Int64, l::Int64, a_d::CuArray{ComplexF64}, b_d::CuArray{ComplexF64},
                              a_sum::CuArray{ComplexF64}, b_sum::CuArray{ComplexF64},aux_a_gpu::CuArray{ComplexF64})# buffers)
    #
    #   Compute a RLWE encruption of Tr_j(μ) given a RLWE encryption
    #   of μ and the Key Switching keys from s(X^d) to s(X)
    #
    CUDA.map!(x -> ComplexF64(x, 0.0), a_sum, A_batch)
    CUDA.map!(x -> ComplexF64(x, 0.0), b_sum, B_batch)

    for k=1:(t-1)
        d = k*t^j + 1

        A_batch_prime = power_gpu(A_batch, d)
        B_batch_prime = power_gpu(B_batch, d)
        
        fill!(a_d, 0.0)
        fill!(b_d, 0.0)
        gpu_batch_rlwe_key_switch(A_batch_prime, B_batch_prime, a_d, b_d, array_FFT_ksk_s_d_to_s[d],D,l, aux_a_gpu)

        a_sum .+= a_d
        b_sum .+= b_d
    end
    #
    CUDA.map!(real, A_batch, a_sum)
    CUDA.map!(real, B_batch, b_sum)

    return
end
function gpu_batch_partial_trace!(A_batch::CuArray{Float64}, B_batch::CuArray{Float64}, j::Int64,
                              array_FFT_ksk_s_d_to_s::Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}},
                              D::Int64, l::Int64, a_d::CuArray{ComplexF64}, b_d::CuArray{ComplexF64},
                              a_sum::CuArray{ComplexF64}, b_sum::CuArray{ComplexF64},
                              a_out::CuArray{Float64},b_out::CuArray{Float64},
                              aux_a_gpu::CuArray{ComplexF64})
    #
    #   Compute a RLWE encruption of Tr_j(μ) given a RLWE encryption
    #   of μ and the Key Switching keys from s(X^d) to s(X)
    #
    CUDA.map!(x -> ComplexF64(x, 0.0), a_sum, A_batch)
    CUDA.map!(x -> ComplexF64(x, 0.0), b_sum, B_batch)

    a_tilted = A_batch
    b_tilted = B_batch

    for k=1:(t-1)
        d = k*t^j + 1

        A_batch_prime = power_gpu(a_tilted, d)
        B_batch_prime = power_gpu(b_tilted, d)
        
        fill!(a_d, 0.0)
        fill!(b_d, 0.0)
        gpu_batch_rlwe_key_switch(A_batch_prime, B_batch_prime, a_d, b_d, array_FFT_ksk_s_d_to_s[d],D,l, aux_a_gpu)

        a_sum .+= a_d
        b_sum .+= b_d
    end
    #
    CUDA.map!(real, a_out, a_sum)
    CUDA.map!(real, b_out, b_sum)

    return
end
function gpu_batch_partial_trace_0!(A_batch::CuArray{Float64}, B_batch::CuArray{Float64},
                              array_FFT_ksk_s_d_to_s::Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}},
                              g::Int64, D::Int64, l::Int64,
                              a_d::CuArray{ComplexF64}, b_d::CuArray{ComplexF64},
                              a_sum::CuArray{ComplexF64}, b_sum::CuArray{ComplexF64},
                              a_tilted::CuArray{Float64}, b_tilted::CuArray{Float64},
                              aux_a_gpu::CuArray{ComplexF64}# buffers
                              )
    #
    #   Compute a RLWE encruption of Tr_j(μ) given a RLWE encryption
    #   of μ and the Key Switching keys from s(X^d) to s(X)
    #
    copyto!(a_tilted, A_batch)
    copyto!(b_tilted, B_batch)
    π = t-1
    aux = 2
    while π > 1
        while !(mod(π, aux)==0)
            aux = nextprime(aux+1)
        end
        π = π ÷ aux
        #a_sum = complex.(a_tilted)
        #b_sum = complex.(b_tilted)
        CUDA.map!(x -> ComplexF64(x, 0.0), a_sum, a_tilted)
        CUDA.map!(x -> ComplexF64(x, 0.0), b_sum, b_tilted)
        #
        for k=1:(aux-1)
            d = powermod(g, k*π , M)
            #
            A_batch_prime = power_gpu(a_tilted, d)
            B_batch_prime = power_gpu(b_tilted, d)
            #
            fill!(a_d, 0.0)
            fill!(b_d, 0.0)
            gpu_batch_rlwe_key_switch(A_batch_prime, B_batch_prime, a_d, b_d, array_FFT_ksk_s_d_to_s[d],D,l, aux_a_gpu)
            #
            a_sum .+= a_d
            b_sum .+= b_d
        end
        #
        CUDA.map!(real, a_tilted, a_sum)
        CUDA.map!(real, b_tilted, b_sum)
    end
    #
    return
end

function kernel_partial_trace0!(
    a_out, a_pows_RLWE, M::Int, N::Int, t::Int, M_over_t::Int, M_inv::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= N && m <= M
        acc = 0.0
        for j in 1:(t - 1)
            idx1 = mod(-(i - 1)* j, M)
            idx2 = mod(-j * (N + mod(i - 1, M_over_t)), M)
            acc += a_pows_RLWE[mod(m - idx1 - 1, M) + 1, j]
            acc -= a_pows_RLWE[mod(m - idx2 - 1, M) + 1, j]
        end
        a_out[m, i] =  acc
    end
    return
end
function kernel_partial_trace_K2toK0!(
    a_out, a_pows_RLWE, M::Int, N::Int, t::Int, M_over_t::Int, M_inv::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= N && m <= M
        acc = 0.0
        for j in 1:(t - 1)
            for k=0(t-1)
                pow = j*(k*t+1)
                idx1 = mod(-(i - 1)* pow, M)
                idx2 = mod(-pow * (N + mod(i - 1, M_over_t)), M)
                acc += a_pows_RLWE[mod(m - idx1 - 1, M) + 1, j+k*(t-1)]
                acc -= a_pows_RLWE[mod(m - idx2 - 1, M) + 1, j+k*(t-1)]
            end
        end
        a_out[m, i] = acc
    end
    return
end

function gpu_fast_partial_trace_0_fused!(
    a_out::CuArray{Float64,2}, 
    a_pows_RLWE::CuArray{Float64,2}
)
    threads = (16, 16)
    blocks = (cld(N, threads[1]), cld(M, threads[2]))

    @cuda threads=threads blocks=blocks kernel_partial_trace0!(a_out, a_pows_RLWE, M, N, t, M_over_t, M_inv)
end

function gpu_fast_partial_trace_K2_to_K0_fused!(
    a_out::CuArray{Float64,2}, 
    a_pows_RLWE::CuArray{Float64,2}
)
    threads = (16, 16)
    blocks = (cld(N, threads[1]), cld(M, threads[2]))

    @cuda threads=threads blocks=blocks kernel_partial_trace_K2toK0!(a_out, a_pows_RLWE, M, N, t, M_over_t, M_inv)
end