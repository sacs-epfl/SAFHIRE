function multi_threaded_rlwe_conv(
                a_rlwe_in_unpacked::AbstractArray{Float64}, 
                b_rlwe_in_unpacked::AbstractArray{Float64}, 
                kernel::Array{Float64, 4}, 
                stride::Int, 
                padding::Int
                )

    a_rlwe_out_unpacked = conv(a_rlwe_in_unpacked, kernel; stride=(stride, stride), pad=(padding, padding))
    b_rlwe_out_unpacked = conv(b_rlwe_in_unpacked, kernel; stride=(stride, stride), pad=(padding, padding))


    return a_rlwe_out_unpacked, b_rlwe_out_unpacked
end

function multi_threaded_partial_extract_trace!(a_rlwe_in_unpacked::AbstractArray{Float64}, b_rlwe_in_unpacked::AbstractArray{Float64}, 
                        a_rlwe::Array{Float64}, b_rlwe::Array{Float64},# (M, t-1, N_chunks)
                        input_size::Int,
                        unshuffle::Permutation;
                        extract_level=1::Int)  # (H, W, C_in, B)
    ##
    num_chunks = size(a_rlwe)[3]

    for chunk_idx=1:num_chunks
        # Copy buffers to output place holder
        start_idx = (chunk_idx-1) * N + 1
        end_idx = min(input_size, N*chunk_idx)
        # Perform Partial Trace Extract
        if extract_level==0
            @views l2_blocked_extract_0!(a_rlwe_in_unpacked[:, start_idx:end_idx], a_rlwe[:,:, chunk_idx])
            @views l2_blocked_extract_0!(b_rlwe_in_unpacked[:, start_idx:end_idx], b_rlwe[:,:, chunk_idx])
        elseif extract_level==1
            @views l2_blocked_trace_0!(a_rlwe_in_unpacked[:, start_idx:end_idx], a_rlwe[:,:, chunk_idx])
            @views l2_blocked_trace_0!(b_rlwe_in_unpacked[:, start_idx:end_idx], b_rlwe[:,:, chunk_idx])
        elseif extract_level==2
            @views l2_blocked_trace!(a_rlwe_in_unpacked[:, start_idx:end_idx], a_rlwe[:,:, chunk_idx])
            @views l2_blocked_trace!(b_rlwe_in_unpacked[:, start_idx:end_idx], b_rlwe[:,:, chunk_idx])
        else
            error("Unsuported Partial Extract Trace Level")
        end
    end
end

function shuffle_output(A::AbstractArray{Float64}, shuffle::Permutation)
    M, output_size = size(A)
    A_shuffled = Array{Float64}(undef, M, output_size)
    Threads.@threads for b = 1:output_size
        @inbounds @simd for m=1:M
            A_shuffled[m, b] = A[m, shuffle(b)]
        end
    end
    return A_shuffled
end

function unshuffle_input(A::AbstractArray{Float64}, unshuffle::Permutation)
    M, input_size = size(A)
    A_unshuffled = Array{Float64}(undef, M, input_size)
    Threads.@threads for b = 1:input_size
        @inbounds @simd for m=1:M
            A_unshuffled[m, unshuffle(b)] = A[m, b]
        end
    end
    return A_unshuffled
end

function multi_threaded_server_forward_conv_layer(
        a_rlwe_in::Array{Float64}, 
        b_rlwe_in::Array{Float64},
        a_rlwe_in_unpacked::AbstractArray{Float64},
        b_rlwe_in_unpacked::AbstractArray{Float64},
        a_rlwe_residual::Ref, # Buffers for residual connections
        b_rlwe_residual::Ref, # Buffers for residual connections
        input_shape::Tuple{Int, Int, Int, Int},
        kernel::Array{Float64, 4},
        stride::Int,
        padding::Int;
        unshuffle=nothing::Union{Permutation, Nothing},
        shuffle=nothing::Union{Permutation, Nothing},
        extract_level=1::Int, # 1 for fast_partial_trace_0, 2 for fast_partial_trace_1_comp_0
        tag=nothing::Union{String, Nothing}, # Tag of layer operation
        projection_kernel=nothing::Union{Array{Float64, 4}, Nothing}
        )
    
    H, W, C_in, B = input_shape
    input_size = prod(input_shape)
    KH, KW, C_in_k, C_out = size(kernel)
    @assert C_in == C_in_k "Input/output channels must match"
    
    t_extract = @elapsed begin
        # Initialize Input RLWEs
        multi_threaded_fill_zeros!(a_rlwe_in_unpacked)
        multi_threaded_fill_zeros!(b_rlwe_in_unpacked)

        # If unshuffle is not provided, use identity permutation
        if isnothing(unshuffle)
            unshuffle = Permutation(prod(input_shape))
        end

        # Perform Partial Trace Extract
        multi_threaded_partial_extract_trace!(a_rlwe_in_unpacked, b_rlwe_in_unpacked,
                            a_rlwe_in,b_rlwe_in, input_size, 
                            unshuffle, extract_level=extract_level)
        # unshuffle ciphertexts
        a_rlwe_in_unpacked = unshuffle_input(a_rlwe_in_unpacked, unshuffle)
        b_rlwe_in_unpacked = unshuffle_input(b_rlwe_in_unpacked, unshuffle)
    end

    # Save the unpacked RLWEs for residual connections
    t_conv = @elapsed begin
        a_rlwe_in_unpacked_T = Matrix{Float64}(undef, input_size, M)
        b_rlwe_in_unpacked_T = Matrix{Float64}(undef, input_size, M)

        simd_blocked_transpose!(a_rlwe_in_unpacked_T,a_rlwe_in_unpacked)
        simd_blocked_transpose!(b_rlwe_in_unpacked_T,b_rlwe_in_unpacked)

        a_rlwe_in_unpacked_T = reshape(a_rlwe_in_unpacked_T, (H, W, C_in,M*B))
        b_rlwe_in_unpacked_T = reshape(b_rlwe_in_unpacked_T, (H, W, C_in,M*B))

        if tag == "residual_in"
            a_rlwe_residual[] = a_rlwe_in_unpacked_T
            b_rlwe_residual[] = b_rlwe_in_unpacked_T
        elseif tag == "residual_projection"
            # Perform projection
            a_rlwe_residual[], b_rlwe_residual[] = multi_threaded_rlwe_conv(a_rlwe_in_unpacked_T, b_rlwe_in_unpacked_T, projection_kernel, 2, 0)
        end
        # Perform convolution
        if tag != "avg_pool"
            a_rlwe_out_unpacked_T, b_rlwe_out_unpacked_T = multi_threaded_rlwe_conv(a_rlwe_in_unpacked_T, b_rlwe_in_unpacked_T, kernel, stride, padding)
        else
            # For average pooling, we use a depthwise convolution with kernel size equal to input size
            a_rlwe_out_unpacked_T = sum(a_rlwe_in_unpacked_T, dims=(1,2)) # Sum over H and W dimensions
            b_rlwe_out_unpacked_T = sum(b_rlwe_in_unpacked_T, dims=(1,2)) # Sum over H and W dimensions
        end

        # Add residual connection if needed
        if tag == "residual_add"
            a_rlwe_out_unpacked_T .+= a_rlwe_residual[]
            b_rlwe_out_unpacked_T .+= b_rlwe_residual[]
        end

        # Transpose back for packing

        H_out, W_out, C_out, BM = size(a_rlwe_out_unpacked_T)
        output_size =  BM * H_out * W_out * C_out ÷ M

        a_rlwe_out_unpacked_T = reshape(a_rlwe_out_unpacked_T, (output_size, M))
        b_rlwe_out_unpacked_T = reshape(b_rlwe_out_unpacked_T, (output_size, M))

        a_rlwe_out_unpacked = Matrix{Float64}(undef, M, output_size)
        b_rlwe_out_unpacked = Matrix{Float64}(undef, M, output_size)
        
        simd_blocked_transpose!(a_rlwe_out_unpacked,a_rlwe_out_unpacked_T)
        simd_blocked_transpose!(b_rlwe_out_unpacked,b_rlwe_out_unpacked_T)
    end

    ### Perform Packing
    t_pack = @elapsed begin

    # Compute number of packing batches
    number_of_packing_batches = output_size ÷ ((t^β * N)÷M ) + 1

    # If shuffle is not provided, use identity permutation
    if isnothing(shuffle)
        shuffle = Permutation(output_size)
    end
    
    # Shuffle output
    a_rlwe_out_unpacked = shuffle_output(a_rlwe_out_unpacked, shuffle)
    b_rlwe_out_unpacked = shuffle_output(b_rlwe_out_unpacked, shuffle)

    # Pack
    a_rlwe_out,b_rlwe_out = layer_optimal_fast_packing(
        a_rlwe_out_unpacked, 
        b_rlwe_out_unpacked,
        β, array_FFT_ksk_s_prime_to_s, 
        g, D, l; 
        start_trace=extract_level)
    end 

    println("t_pack : ", t_pack)
    # Return the output RLWEs
    return a_rlwe_out,b_rlwe_out, t_extract, t_conv, t_pack
end
