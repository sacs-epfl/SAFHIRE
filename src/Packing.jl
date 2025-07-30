
"""
    optimal_fast_packing(a_0::Array{Float64}, b_0::Matrix{Float64}, β::Int64, 
                            a::Array{Float64, 3}, b::Array{Float64, 3},
                            array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                            g::Int64, D::Int64, l::Int64; start_trace == 0)

    Computes the optimal fast packing of RLWE ciphertexts for a given β.
"""
function optimal_fast_packing(a_0::Array{Float64}, b_0::Matrix{Float64}, β::Int64, 
                            a::Array{Float64, 3}, b::Array{Float64, 3},
                            array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                            g::Int64, D::Int64, l::Int64; start_trace=0::Int64)
    if β==1
        return(zeros(M), zeros(M))
    end

    # Define buffers for inplace functions
    a_tilded = zeros(M)
    b_tilded = zeros(M)
    buffer_a_sum = zeros(Float64, M)
    buffer_b_sum = zeros(Float64, M)
    buffer_a_d = zeros(Float64, M)
    buffer_b_d = zeros(Float64, M)

    for i=1:(t^(β-1))
        a_i = zeros(M)
        b_i = zeros(M)
        for j=0:(t-2)
            @views copyto!(a_tilded, a_0[:,i+j*t^(β-1)])
            @views copyto!(b_tilded, b_0[:,i+j*t^(β-1)])
            # If strat trace is 0 we compute the first partial trace
            if start_trace == 0
                partial_trace_0!(a_tilded, b_tilded,
                                array_FFT_ksk_s_prime_to_s,
                                g, D, l,
                                buffer_a_sum, buffer_b_sum,
                                buffer_a_d, buffer_b_d)
            end
            rotate_poly!(a_i, a_tilded, j*M÷(t)+1)
            rotate_poly!(b_i, b_tilded, j*M÷(t)+1)
        end
        @views a[:, i, 1] = a_i
        @views b[:, i, 1] = b_i
    end

    # Now we compute the partial traces for j=2 to β-1
    for j=2:(β-1)
        for i=1:t^(β-j)
            a_i = zeros(M)
            b_i = zeros(M)
            for i_β = 0:(t-1)
                @views copyto!(a_tilded, a[:,i+i_β*t^(β-j),j-1])
                @views copyto!(b_tilded, b[:,i+i_β*t^(β-j),j-1])
                if j > start_trace
                    partial_trace!(a_tilded, b_tilded, j-1,
                                array_FFT_ksk_s_prime_to_s,
                                D, l,
                                buffer_a_sum, buffer_b_sum,
                                buffer_a_d, buffer_b_d)
                end
                rotate_poly!(a_i, a_tilded, i_β*M÷(t^j)+1)
                rotate_poly!(b_i, b_tilded, i_β*M÷(t^j)+1)
            end
            @views copyto!(a[:, i, j], a_i)
            @views copyto!(b[:, i, j], b_i)
        end
    end

    a_β = zeros(M)
    b_β = zeros(M)
    for i=0:t-1
        aux_a, aux_b = partial_trace((a[:,i+1,β-1], b[:,i+1,β-1]), β-1, array_FFT_ksk_s_prime_to_s, D, l)
        rotate_poly!(a_β, aux_a, i*M÷(t^β)+1)
        rotate_poly!(b_β, aux_b, i*M÷(t^β)+1)
    end

    for index = β:(α-1)
        a_β, b_β = partial_trace((a_β, b_β), index , array_FFT_ksk_s_prime_to_s, D, l)
    end

    return a_β, b_β
end

function transform_poly_cache_friendly!(
    a_new::Matrix{T}, b_new::Matrix{T},
    a_0::Matrix{T}, b_0::Matrix{T},
    output_size::Int, new_output_size::Int, pck_size::Int
) where {T}

    tb = t^(β - 1)

    Threads.@threads for pck_idx = 1:div(new_output_size, tb)
        new_pck_idx = (pck_idx - 1) * tb

        for j = 0:(t - 2)
            idx1 = j * M ÷ t

            # access linear_ind in ascending order
            @inbounds for i = 1:tb
                linear_ind = i + j * tb + (pck_idx - 1) * pck_size
                if linear_ind > output_size
                    continue
                end

                col = new_pck_idx + i

                @simd for m = 1:idx1
                    row1 = M + (m - idx1)
                    a_new[m, col] += a_0[row1, linear_ind]
                    b_new[m, col] += b_0[row1, linear_ind]
                end

                @simd for m = idx1 + 1:M
                    row1 = m - idx1
                    a_new[m, col] += a_0[row1, linear_ind]
                    b_new[m, col] += b_0[row1, linear_ind]
                end
            end
        end
    end
end


"""
Performs an optimized packing operation for a given layer using the provided input arrays and parameter.

# Arguments
- `a_0::Array{Float64}`: A two-dimensional array respresenting the first part of the mask of the RLWEs to pack.
- `b_0::Matrix{Float64}`: A two-dimensional array representing the second part of the mask of the RLWEs to pack.
- `β::Int64`: An integer parameter controlling the packing strategy or the number of elements to pack.
- `array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}}`: An array containing precomputed FFT keys for the key switching operations.
- `g::Int64`: A generator used in the packing operation.
- `D::Int64`: Scaling factor in the key switching operation.
- `l::Int64`: Depth in the key switching operation.

# Returns
- Packed RLWEs.
"""
function layer_optimal_fast_packing(a_0::Array{Float64}, b_0::Matrix{Float64}, β::Int64,
                            array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                            g::Int64, D::Int64, l::Int64; start_trace=0::Int64)
    b_size = cpu_batch_size 
    output_size = size(a_0, 2)

    # Define temporary buffers
    a_sum = Array{Float64}(undef, M, cpu_batch_size)
    b_sum = Array{Float64}(undef, M, cpu_batch_size)
    a_d = Array{Float64}(undef, M, cpu_batch_size)
    b_d = Array{Float64}(undef, M, cpu_batch_size)
    a_buf = Array{ComplexF64}(undef, M, cpu_batch_size)
    b_buf = Array{ComplexF64}(undef, M, cpu_batch_size)
    a_aux = Array{ComplexF64}(undef, M, cpu_batch_size,l)

    # If strat trace is 0 Apply Trace 0
    if start_trace == 0
        batch_number = ceil(Int, output_size / b_size)
        for batch_idx = 1:batch_number
            start_idx = (batch_idx-1)*b_size + 1
            end_idx = min(output_size, batch_idx*b_size)
            @views batch_partial_trace_0!(
                a_0[:,start_idx:end_idx], b_0[:,start_idx:end_idx], 
                array_FFT_ksk_s_prime_to_s, 
                g ,D ,l,
                a_sum,b_sum,
                a_d,b_d
                )
        end
    end

    # Perform rotate add 

    pck_size = (t^β * N ÷ M)
    nb_pck = ceil(Int, output_size / pck_size)

    new_output_size = nb_pck * t^(β-1)

    a_new = Array{Float64}(undef, M, new_output_size)
    b_new = Array{Float64}(undef, M, new_output_size)
    multi_threaded_fill_zeros!(a_new)
    multi_threaded_fill_zeros!(b_new)

    transform_poly_cache_friendly!(a_new, b_new, a_0, b_0, output_size, new_output_size, pck_size)

    if start_trace < 2
        a_in = real_to_torus(a_new)
        b_in = real_to_torus(b_new)
    else
        a_in = a_new
        b_in = b_new
    end

    # Perform partial traces
    for j=2:β
        # Inplace Batch Partial Traces
        output_size = size(a_in, 2)
        if j > start_trace
            batch_number = ceil(Int, output_size / b_size)
            for batch_idx = 1:batch_number
                start_idx = (batch_idx-1)*b_size + 1
                end_idx = min(output_size, batch_idx*b_size)
                if end_idx-start_idx+1 == b_size
                    @views batch_partial_trace!(
                        a_in[:,start_idx:end_idx], b_in[:,start_idx:end_idx], 
                        j-1 ,FFT_automorphosim_ksk, 
                        D ,l,
                        a_sum,b_sum,
                        a_d,b_d,
                        a_buf, b_buf,
                        a_aux
                        )
                else
                    last_b_size = end_idx-start_idx+1
                    a_sum_b = Array{Float64}(undef, M, last_b_size)
                    b_sum_b = Array{Float64}(undef, M, last_b_size)
                    a_d_b = Array{Float64}(undef, M, last_b_size)
                    b_d_b = Array{Float64}(undef, M, last_b_size)
                    a_buf_b = Array{ComplexF64}(undef, M, last_b_size)
                    b_buf_b = Array{ComplexF64}(undef, M, last_b_size)
                    a_aux_b = Array{ComplexF64}(undef, M, last_b_size,l)
                    @views batch_partial_trace!(
                        a_in[:,start_idx:end_idx], b_in[:,start_idx:end_idx], 
                        j-1 ,FFT_automorphosim_ksk, 
                        D ,l,
                        a_sum_b,b_sum_b,
                        a_d_b,b_d_b,
                        a_buf_b, b_buf_b,
                        a_aux_b
                        )
                end     
            end
        end

        # Perform Add Rotate
        new_output_size = t^(β-j) * nb_pck
        pck_size =  t^(β-j+1)


        a_new = Array{Float64}(undef, M, new_output_size)
        b_new = Array{Float64}(undef, M, new_output_size)
        multi_threaded_fill_zeros!(a_new)
        multi_threaded_fill_zeros!(b_new)

        for pck_idx=1:nb_pck
            for i_β = 0:(t-1)
                Threads.@threads for i=1:t^(β-j)
                    linear_ind = (pck_idx-1)*pck_size + i_β*t^(β-j) + i
                    if linear_ind > output_size
                        continue
                    end
                    new_pck_idx = (pck_idx-1) * t^(β-j)
                    @views rotate_poly!(a_new[:,new_pck_idx+i], a_in[:, linear_ind], i_β*M÷(t^j)+1)
                    @views rotate_poly!(b_new[:,new_pck_idx+i], b_in[:, linear_ind], i_β*M÷(t^j)+1)
                end
            end
        end

        a_in = real_to_torus(a_new)
        b_in = real_to_torus(b_new)
    end

    for ind = β:(α-1)
        output_size = size(a_in, 2)
        batch_number = ceil(Int, output_size / b_size)
        for batch_idx = 1:batch_number
            start_idx = (batch_idx-1)*b_size + 1
            end_idx = min(output_size, batch_idx*b_size)
            if end_idx-start_idx+1 == b_size
                @views batch_partial_trace!(
                    a_in[:,start_idx:end_idx], b_in[:,start_idx:end_idx], 
                    ind ,FFT_automorphosim_ksk, 
                    D ,l,
                    a_sum,b_sum,
                    a_d,b_d,
                    a_buf, b_buf,
                    a_aux
                    )
            else
                last_b_size = end_idx-start_idx+1
                a_sum_b = Array{Float64}(undef, M, last_b_size)
                b_sum_b = Array{Float64}(undef, M, last_b_size)
                a_d_b = Array{Float64}(undef, M, last_b_size)
                b_d_b = Array{Float64}(undef, M, last_b_size)
                a_buf_b = Array{ComplexF64}(undef, M, last_b_size)
                b_buf_b = Array{ComplexF64}(undef, M, last_b_size)
                a_aux_b = Array{ComplexF64}(undef, M, last_b_size,l)
                @views batch_partial_trace!(
                    a_in[:,start_idx:end_idx], b_in[:,start_idx:end_idx], 
                    ind ,FFT_automorphosim_ksk, 
                    D ,l,
                    a_sum_b,b_sum_b,
                    a_d_b,b_d_b,
                    a_buf_b, b_buf_b,
                    a_aux_b
                    )
            end    
        end
    end

    return a_in, b_in
end