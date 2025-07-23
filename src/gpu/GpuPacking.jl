function gpu_optimal_fast_packing(A_batch_vect::CuArray{Float64}, B_batch_vec::CuArray{Float64},β::Int64,
                            array_FFT_ksk_s_d_to_s::Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}},
                            g::Int64, D::Int64, l::Int64;start_trace=1::Int64)
    #
    #   Batch Fast Packing of RLWE ciphertexts ie N_pack number of batches.
    #

    # Preallocate output buffers
    a = CuArray{Float64}(undef,M, batch_size, t^(β-1),β-1)
    b = CuArray{Float64}(undef,M, batch_size, t^(β-1),β-1)

    # Preallocate buffers for Partial traces
    a_i = CUDA.zeros(Float64, M,batch_size)
    b_i = CUDA.zeros(Float64, M,batch_size)
    a_d, b_d = CUDA.zeros(ComplexF64, M, batch_size), CUDA.zeros(ComplexF64, M, batch_size)

    a_sum = CuArray{ComplexF64}(undef, M, batch_size)
    b_sum = CuArray{ComplexF64}(undef, M, batch_size)

    a_tilted = CuArray{Float64}(undef, M, batch_size)
    b_tilted = CuArray{Float64}(undef, M, batch_size)

    aux_a_gpu = CuArray{ComplexF64}(undef, M, batch_size, l)

    for i=1:(t^(β-1))
        fill!(a_i, 0.0)
        fill!(b_i, 0.0)
        for j=0:(t-2)
            if start_trace==0
                @views gpu_batch_partial_trace_0!(
                    A_batch_vect[:,:,i+j*t^(β-1)], 
                    B_batch_vec[:,:,i+j*t^(β-1)], 
                    array_FFT_ksk_s_d_to_s, 
                    g, D, l, 
                    a_d, b_d, 
                    a_sum, b_sum, 
                    a_tilted, b_tilted, 
                    aux_a_gpu
                    )
            else
                @views copyto!(a_tilted, A_batch_vect[:,:,i+j*t^(β-1)])
                @views copyto!(b_tilted, B_batch_vec[:,:,i+j*t^(β-1)])
            end
            batch_rotate_poly_gpu!(a_i,a_tilted, j* M÷t)
            batch_rotate_poly_gpu!(b_i, b_tilted, j* M÷t)
        end
        @views copyto!(a[:,:,i,1], a_i)
        @views copyto!(b[:,:,i,1], b_i)
    end
    #end

    # Now we compute the partial traces for j=2 to β-1
    for j=2:(β-1)
        for i=1:t^(β-j)
            fill!(a_i, 0.0)
            fill!(b_i, 0.0)
            for i_β = 0:(t-1)
                if j > start_trace
                    gpu_batch_partial_trace!(
                        a[:,:,i+i_β*t^(β-j),j-1], 
                        b[:,:,i+i_β*t^(β-j),j-1], 
                        j-1 , array_FFT_ksk_s_d_to_s, 
                        D, l, a_d, b_d, 
                        a_sum,b_sum, 
                        a_tilted, b_tilted,
                        aux_a_gpu
                        )
                else
                    @views copyto!(a_tilted, a[:,:,i+i_β*t^(β-j),j-1])
                    @views copyto!(b_tilted, b[:,:,i+i_β*t^(β-j),j-1])
                end
                batch_rotate_poly_gpu!(a_i, a_tilted, i_β*M÷(t^j))
                batch_rotate_poly_gpu!(b_i, b_tilted, i_β*M÷(t^j))
            end
            @views copyto!(a[:,:,i,j], a_i)
            @views copyto!(b[:,:,i,j], b_i)
        end
    end

    a_β = CUDA.zeros(Float64, M, batch_size)
    b_β = CUDA.zeros(Float64, M, batch_size)
    for i=0:t-1
        @views gpu_batch_partial_trace!(
            a[:,:,i+1, β-1], 
            b[:,:,i+1, β-1],
            β-1 , array_FFT_ksk_s_d_to_s, 
            D, l, a_d, b_d, 
            a_sum,b_sum, 
            a_tilted, b_tilted,
            aux_a_gpu
            )
        batch_rotate_poly_gpu!(a_β, a_tilted, i*M÷(t^β))
        batch_rotate_poly_gpu!(b_β, b_tilted, i*M÷(t^β))
    end

    for ind = β:(α-1)
        gpu_batch_partial_trace!(
            a_β, 
            b_β,
            ind , array_FFT_ksk_s_d_to_s, 
            D, l, a_d, b_d, 
            a_sum,b_sum, 
            aux_a_gpu
            )
    end
    #
    return a_β, b_β
end