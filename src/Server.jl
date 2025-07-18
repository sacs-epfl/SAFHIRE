using Permutations, NNlib

function partial_extract_trace!(a_rlwe_in_unpacked::Array{Float64}, b_rlwe_in_unpacked::Array{Float64}, 
                        a_rlwe::Array{Float64}, b_rlwe::Array{Float64},# (M, t-1, N_chunks)
                        input_shape::Tuple{Int, Int, Int, Int},
                        unshuffle::Permutation;
                        extract_level=1::Int)  # (H, W, C_in, B)
    ##
    num_chunks = size(a_rlwe)[3]

    # Compute indeces to go from linear to Cartesian
    indices = CartesianIndices(input_shape)
    in_dim = prod(input_shape)

    a_out = zeros(M)
    b_out = zeros(M)
    a_rlwe_pows = Array{Float64}(undef, M, (t-1)*t^(extract_level-1))
    b_rlwe_pows = Array{Float64}(undef, M, (t-1)*t^(extract_level-1))

    for chunk_idx=1:num_chunks
        @views copyto!(a_rlwe_pows, a_rlwe[:,:, chunk_idx])
        @views copyto!(b_rlwe_pows, b_rlwe[:,:, chunk_idx])
        for i=1:N
            ###############################################
            ### Insert Permutation here for stronger DP ###
            ###############################################
            linear_ind = (chunk_idx-1) * N + i

            if linear_ind > in_dim
                continue
            end

            linear_ind = unshuffle(linear_ind)

            # Get CartesianIndices
            h,w,c,b = Tuple(indices[linear_ind])

            # Perform Partial Trace Extract
            if extract_level==1
                fast_partial_trace_0!(a_out, b_out, a_rlwe_pows,b_rlwe_pows,i-1)
            elseif extract_level==2
                fast_partial_trace_1_comp_0!(a_out, b_out, a_rlwe_pows,b_rlwe_pows,i-1)
            else
                error("Unsuported Partial Extract Trace Level")
            end

            @views copyto!(a_rlwe_in_unpacked[(b-1)*M+1 : b*M,h,w,c], a_out)
            @views copyto!(b_rlwe_in_unpacked[(b-1)*M+1 : b*M,h,w,c], b_out)
        end
    end
end

function partial_extract_trace!(a_rlwe_in_unpacked::Array{Float64}, b_rlwe_in_unpacked::Array{Float64}, 
                        a_rlwe::Array{Float64}, b_rlwe::Array{Float64},# (M, t-1, N_chunks)
                        input_dim::Int64,
                        unshuffle::Permutation;
                        extract_level=1::Int)  # (H, W, C_in, B)
    ##
    num_chunks = size(a_rlwe)[3]

    a_out = zeros(M)
    b_out = zeros(M)
    a_rlwe_pows = Array{Float64}(undef, M, (t-1)*t^(extract_level-1))
    b_rlwe_pows = Array{Float64}(undef, M, (t-1)*t^(extract_level-1))

    for chunk_idx=1:num_chunks
        @views copyto!(a_rlwe_pows, a_rlwe[:,:, chunk_idx])
        @views copyto!(b_rlwe_pows, b_rlwe[:,:, chunk_idx])
        for i=1:N
            ###############################################
            ### Insert Permutation here for stronger DP ###
            ###############################################
            linear_ind = (chunk_idx-1) * N + i

            if linear_ind > input_dim
                continue
            end

            linear_ind = unshuffle(linear_ind)

            # Perform Partial Trace Extract
            if extract_level==1
                fast_partial_trace_0!(a_out, b_out, a_rlwe_pows,b_rlwe_pows,i-1)
            elseif extract_level==2
                fast_partial_trace_1_comp_0!(a_out, b_out, a_rlwe_pows,b_rlwe_pows,i-1)
            else
                error("Unsuported Partial Extract Trace Level")
            end

            @views copyto!(a_rlwe_in_unpacked[:,linear_ind], a_out)
            @views copyto!(b_rlwe_in_unpacked[:,linear_ind], b_out)
        end
    end
end

function rlwe_conv(
                a_rlwe_in_unpacked::Array{Float64}, 
                b_rlwe_in_unpacked::Array{Float64}, 
                kernel::Array{Float64, 4}, 
                stride::Int, 
                padding::Int
                )

    # Permute dims for convolution
    a_rlwe_in_unpacked = permutedims(a_rlwe_in_unpacked, (2,3,4,1))
    b_rlwe_in_unpacked = permutedims(b_rlwe_in_unpacked, (2,3,4,1))


    a_rlwe_out_unpacked = conv(a_rlwe_in_unpacked, kernel; stride=(stride, stride), pad=(padding, padding))
    b_rlwe_out_unpacked = conv(b_rlwe_in_unpacked, kernel; stride=(stride, stride), pad=(padding, padding))

    # Permute back dims to allow for non strided slicing 
    a_rlwe_out_unpacked = permutedims(a_rlwe_out_unpacked, (4,1,2,3))
    b_rlwe_out_unpacked = permutedims(b_rlwe_out_unpacked, (4,1,2,3))

    return a_rlwe_out_unpacked, b_rlwe_out_unpacked
end

function pack_output!(a_rlwe_out::Array{Float64}, b_rlwe_out::Array{Float64}, 
                    a_rlwe_out_unpacked::Array{Float64}, b_rlwe_out_unpacked::Array{Float64},
                    output_shape::Tuple{Int, Int, Int, Int}, # (H_out, W_out, C_out, B_outs)
                    shuffle::Permutation;
                    extract_level=1::Int
                    )

    # Compute number of packing batches
    number_of_packing_batches = size(a_rlwe_out)[2]

    # Compute mapping from flat to 4D indices
    H_out,W_out,C_out,b_out = output_shape
    indices_out = CartesianIndices((H_out, W_out, C_out, b_out ÷ M))
    output_size = length(indices_out)

    # Initialize buffers for packing
    a_buf = Array{Float64}(undef,M, t^(β-1),β-1)
    b_buf = Array{Float64}(undef,M, t^(β-1),β-1)

    a_packing_in = zeros(M,(t^β * N)÷M)
    b_packing_in = zeros(M,(t^β * N)÷M)

    for batch_id = 1:number_of_packing_batches
        # Perform packing
        for i=1:(t^β * N)÷M
            # Check if index is in bound
            out_ind = (batch_id - 1) * ((t^β * N)÷M) + i
            if out_ind > output_size
                break
            end

            # Apply shuffling
            out_ind = shuffle(out_ind)
            # Get CartesianIndices
            h,w,c,b = Tuple(indices_out[out_ind])
            # Perform the dot RLWE Dot product
            @views a_packing_in[:,i] = a_rlwe_out_unpacked[(b-1)*M+1 : b*M,h,w,c]
            @views b_packing_in[:,i] = b_rlwe_out_unpacked[(b-1)*M+1 : b*M,h,w,c]
        end

        a_rlwe_out[:, batch_id], b_rlwe_out[:, batch_id] = optimal_fast_packing(a_packing_in, b_packing_in, β, a_buf, b_buf, array_FFT_ksk_s_prime_to_s, g, D, l; start_trace=extract_level)
    end
end

function pack_output!(a_rlwe_out::Array{Float64}, b_rlwe_out::Array{Float64}, 
                    a_rlwe_out_unpacked::Array{Float64}, b_rlwe_out_unpacked::Array{Float64},
                    output_size::Int, # (H_out, W_out, C_out, B_outs)
                    shuffle::Permutation;
                    extract_level=1::Int
                    )

    # Compute number of packing batches
    number_of_packing_batches = size(a_rlwe_out)[2]

    # Initialize buffers for packing
    a_buf = Array{Float64}(undef,M, t^(β-1),β-1)
    b_buf = Array{Float64}(undef,M, t^(β-1),β-1)

    a_packing_in = zeros(M,(t^β * N)÷M)
    b_packing_in = zeros(M,(t^β * N)÷M)

    for batch_id = 1:number_of_packing_batches
        # Perform packing
        for i=1:(t^β * N)÷M
            # Check if index is in bound
            out_ind = (batch_id - 1) * ((t^β * N)÷M) + i
            if out_ind > output_size
                continue
            end

            # Apply shuffling
            out_ind = shuffle(out_ind)            

            # Perform the dot RLWE Dot product
            @views a_packing_in[:,i] = a_rlwe_out_unpacked[:,out_ind]
            @views b_packing_in[:,i] = b_rlwe_out_unpacked[:,out_ind]
        end

        a_rlwe_out[:, batch_id], b_rlwe_out[:, batch_id] = optimal_fast_packing(a_packing_in, b_packing_in, β, a_buf, b_buf, array_FFT_ksk_s_prime_to_s, g, D, l; start_trace=extract_level)
    end
end

function server_forward_conv_layer(
        a_rlwe_in::Array{Float64}, 
        b_rlwe_in::Array{Float64},
        #a_rlwe_residual::Union{Array{Float64}, Nothing}, # Buffers for residual connections
        #b_rlwe_residual::Union{Array{Float64}, Nothing}, # Buffers for residual connections
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
    KH, KW, C_in_k, C_out = size(kernel)
    @assert C_in == C_in_k "Input/output channels must match"
    
    t_extract = @elapsed begin
    # Initialize Input RLWEs
    a_rlwe_in_unpacked = Array{Float64}(undef, M*B, H, W, C_in)
    b_rlwe_in_unpacked = Array{Float64}(undef, M*B, H, W, C_in)

    # If unshuffle is not provided, use identity permutation
    if isnothing(unshuffle)
        unshuffle = Permutation(prod(input_shape))
    end

    # Perform Partial Trace Extract
    partial_extract_trace!(a_rlwe_in_unpacked, b_rlwe_in_unpacked,
                           a_rlwe_in,b_rlwe_in, input_shape, 
                           unshuffle, extract_level=extract_level
                        )
    end

    # Save the unpacked RLWEs for residual connections
    t_conv = @elapsed begin
    if tag == "residual_in"
        a_rlwe_residual[] = a_rlwe_in_unpacked
        b_rlwe_residual[] = b_rlwe_in_unpacked
    elseif tag == "residual_projection"
        # Perform projection
        a_rlwe_residual[], b_rlwe_residual[] = rlwe_conv(a_rlwe_in_unpacked, b_rlwe_in_unpacked, projection_kernel, 2, 0)
    end
    # Perform convolution
    if tag != "avg_pool"
        a_rlwe_out_unpacked, b_rlwe_out_unpacked = rlwe_conv(a_rlwe_in_unpacked, b_rlwe_in_unpacked, kernel, stride, padding)
    else
        # For average pooling, we use a depthwise convolution with kernel size equal to input size
        a_rlwe_out_unpacked = sum(a_rlwe_in_unpacked, dims=(2,3)) # Sum over H and W dimensions
        b_rlwe_out_unpacked = sum(b_rlwe_in_unpacked, dims=(2,3)) # Sum over H and W dimensions
    end

    # Add residual connection if needed
    if tag == "residual_add"
        a_rlwe_out_unpacked .+= a_rlwe_residual[]
        b_rlwe_out_unpacked .+= b_rlwe_residual[]
    end
    end

    ### Perform Packing
    t_pack = @elapsed begin
    BM, H_out, W_out, C_out = size(a_rlwe_out_unpacked)

    output_size =  BM * H_out * W_out * C_out ÷ M
    output_shape = (H_out, W_out, C_out, BM)

    # Compute number of packing batches
    number_of_packing_batches = output_size ÷ ((t^β * N)÷M ) + 1

    # Preallocate output
    a_rlwe_out = Array{Float64}(undef, M, number_of_packing_batches)
    b_rlwe_out = Array{Float64}(undef, M, number_of_packing_batches)

    # If shuffle is not provided, use identity permutation
    if isnothing(shuffle)
        shuffle = Permutation(output_size)
    end

    # Pack
    pack_output!(a_rlwe_out, b_rlwe_out, a_rlwe_out_unpacked, b_rlwe_out_unpacked, output_shape, shuffle, extract_level=extract_level)
    end
    # Return the output RLWEs
    return a_rlwe_out,b_rlwe_out, t_extract, t_conv, t_pack
end

function server_forward_fc_layer(
        a_rlwe_in::Array{Float64}, 
        b_rlwe_in::Array{Float64},
        weights::Array{Float64, 2}; # Weights for fully connected layer (Input_dim, Output_dim)
        unshuffle=nothing::Union{Permutation, Nothing},
        extract_level=1::Int, # 1 for fast_partial_trace_0, 2 for fast_partial_trace_1_comp_0
    )
    t_extract = @elapsed begin
    # Implementation similar to server_forward_conv_layer but adapted for fully connected layers
    input_size, output_size = size(weights)

    a_rlwe_in_unpacked = Array{Float64}(undef, M, input_size)
    b_rlwe_in_unpacked = Array{Float64}(undef, M, input_size)

    # If unshuffle is not provided, use identity permutation
    if isnothing(unshuffle)
        unshuffle = Permutation(input_size)
    end

    # Perform Partial Trace Extract
    partial_extract_trace!(a_rlwe_in_unpacked, b_rlwe_in_unpacked,
                           a_rlwe_in,b_rlwe_in, input_size, 
                           unshuffle, extract_level=extract_level
                        )
    end
    t_conv = @elapsed begin
    # Perform matrix multiplication
    a_rlwe_out_unpacked = a_rlwe_in_unpacked * weights
    b_rlwe_out_unpacked = b_rlwe_in_unpacked * weights
    end
    # Compute number of packing batches
    number_of_packing_batches = output_size ÷ ((t^β * N)÷M ) + 1

    # Preallocate output
    t_pack = @elapsed begin
    a_rlwe_out = Array{Float64}(undef, M, number_of_packing_batches)
    b_rlwe_out = Array{Float64}(undef, M, number_of_packing_batches)

    shuffle = Permutation(output_size)
    # Pack
    pack_output!(a_rlwe_out, b_rlwe_out, a_rlwe_out_unpacked, b_rlwe_out_unpacked, output_size, shuffle, extract_level=extract_level)
    end
    return a_rlwe_out, b_rlwe_out, t_extract, t_conv, t_pack
end