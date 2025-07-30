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
                        j-1 , CuFFT_automorphism_ksk, 
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
            β-1 , CuFFT_automorphism_ksk, 
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
            ind , CuFFT_automorphism_ksk, 
            D, l, a_d, b_d, 
            a_sum,b_sum, 
            aux_a_gpu
            )
    end
    #
    return a_β, b_β
end

function gpu_layer_optimal_fast_packing(a_0::CuArray{Float64}, b_0::CuArray{Float64}, β::Int64,
                            array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{CuArray{ComplexF64}, CuArray{ComplexF64}}}},
                            g::Int64, D::Int64, l::Int64; start_trace=0::Int64)
    b_size = batch_size 
    output_size = size(a_0, 2)

    # Define temporary buffers
    a_d, b_d = CUDA.zeros(ComplexF64, M, batch_size), CUDA.zeros(ComplexF64, M, batch_size)

    a_sum = CuArray{ComplexF64}(undef, M, batch_size)
    b_sum = CuArray{ComplexF64}(undef, M, batch_size)

    a_tilded = CuArray{Float64}(undef, M, batch_size)
    b_tilded = CuArray{Float64}(undef, M, batch_size)

    aux_a_gpu = CuArray{ComplexF64}(undef, M, batch_size, l)

    # Apply Trace 0
    if start_trace == 0
        batch_number = ceil(Int, output_size / b_size)
        for batch_idx = 1:batch_number
            start_idx = (batch_idx-1)*b_size + 1
            end_idx = min(output_size, batch_idx*b_size)
            @views gpu_batch_partial_trace_0!(
                a_0[:,start_idx:end_idx], b_0[:,start_idx:end_idx], 
                array_FFT_ksk_s_prime_to_s, 
                g ,D ,l,
                a_sum,b_sum,
                a_d,b_d,
                aux_a_gpu
                )
        end
    end

    # Compute sizes
    pck_size = (t^β * N ÷ M)
    nb_pck = ceil(Int, output_size / pck_size)
    new_output_size = nb_pck * t^(β-1) #ceil(Int, output_size / (t-1))

    # Allocate output for rotate add
    a_new = CUDA.zeros(Float64, M, new_output_size)
    b_new = CUDA.zeros(Float64, M, new_output_size)


    # Perform rotate add 
    threads = 256
    blocks = ceil(Int, new_output_size / threads)

    @cuda threads=threads blocks=blocks fused_rotate_pack!(
        a_new, b_new, a_0, b_0, M, t, β, output_size,new_output_size,N
    )

    a_in = a_new
    b_in = b_new
    # Perform partial traces
    for j=2:β
        output_size = size(a_in, 2)
        elapsed = CUDA.@elapsed begin
        if j > start_trace
            batch_number = ceil(Int, output_size / b_size)
            for batch_idx = 1:batch_number
                start_idx = (batch_idx-1)*b_size + 1
                end_idx = min(output_size, batch_idx*b_size)
                len = end_idx - start_idx + 1

                @views copyto!(a_tilded[:, 1:len], a_in[:,start_idx:end_idx])
                @views copyto!(b_tilded[:, 1:len], b_in[:,start_idx:end_idx])

                @views gpu_batch_partial_trace!(
                    a_tilded, b_tilded, 
                    j-1 ,array_FFT_ksk_s_prime_to_s, 
                    D ,l,
                    a_sum,b_sum,
                    a_d,b_d,
                    aux_a_gpu
                    )

                @views copyto!(a_in[:,start_idx:end_idx], a_tilded[:, 1:len])
                @views copyto!(b_in[:,start_idx:end_idx], b_tilded[:, 1:len])
            end
        end
        end
        println("time Trace $(j) : ", elapsed)
        # Perform Add Rotate
        elapsed = CUDA.@elapsed begin
        new_output_size = t^(β-j) * nb_pck
        pck_size =  t^(β-j+1)
        #println("New output site : ", new_output_size)
        a_new = CUDA.zeros(Float64, M, new_output_size)
        b_new = CUDA.zeros(Float64, M, new_output_size)
        
        threads = 256
        blocks = ceil(Int, size(a_new, 2) / threads)

        # Perform add rotate
        @cuda threads=threads blocks=blocks fused_add_rotate!(
            a_new, b_new,
            a_in, b_in,
            M, t, β, j,
            output_size, nb_pck
        )

        a_in = real_to_torus.(a_new)
        b_in = real_to_torus.(b_new)
        end
        println("Rotate $(j) : ", elapsed)
    end

    
    for ind = β:(α-1)
        output_size = size(a_in, 2)
        batch_number = ceil(Int, output_size / b_size)
        for batch_idx = 1:batch_number
            start_idx = (batch_idx-1)*b_size + 1
            end_idx = min(output_size, batch_idx*b_size)
            len = end_idx - start_idx + 1

            @views copyto!(a_tilded[:, 1:len], a_in[:,start_idx:end_idx])
            @views copyto!(b_tilded[:, 1:len], b_in[:,start_idx:end_idx])

            @views gpu_batch_partial_trace!(
                a_tilded, b_tilded, 
                ind ,array_FFT_ksk_s_prime_to_s, 
                D ,l,
                a_sum,b_sum,
                a_d,b_d,
                aux_a_gpu
                )

            @views copyto!(a_in[:,start_idx:end_idx], a_tilded[:, 1:len])
            @views copyto!(b_in[:,start_idx:end_idx], b_tilded[:, 1:len])
        end
    end
    println(size(a_in))

    return a_in, b_in
end