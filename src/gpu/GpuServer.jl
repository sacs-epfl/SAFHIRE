function gpu_partial_extract_trace!(
                        a_rlwe_in_unpacked::CuArray{Float64}, b_rlwe_in_unpacked::CuArray{Float64}, 
                        a_rlwe::CuArray{Float64}, b_rlwe::CuArray{Float64},# (M, t-1, N_chunks)
                        input_size::Int;
                        extract_level=1::Int)  
    # Compute number of chunks
    num_chunks = size(a_rlwe)[3]

    # Preallocate buffer for kernels
    a_out, b_out = CUDA.zeros(Float64, M, N), CUDA.zeros(Float64, M, N)

    for chunk_idx=1:num_chunks
        fill!(a_out, 0.0)
        fill!(b_out, 0.0)

        # Perform Partial Trace Extract
        if extract_level== 0
            @views gpu_l2_blocked_extract_0!(a_out, a_rlwe[:,:, chunk_idx])
            @views gpu_l2_blocked_extract_0!(b_out, b_rlwe[:,:, chunk_idx])
        elseif extract_level==1
            @views gpu_fast_partial_trace_0_fused!(a_out, a_rlwe[:,:, chunk_idx])
            @views gpu_fast_partial_trace_0_fused!(b_out, b_rlwe[:,:, chunk_idx])
        elseif extract_level==2
            @views gpu_fast_partial_trace_K2_to_K0_fused!(a_out, a_rlwe[:,:, chunk_idx])
            @views gpu_fast_partial_trace_K2_to_K0_fused!(b_out, b_rlwe[:,:, chunk_idx])
        else
            error("Unsuported Partial Extract Trace Level")
        end

        # Copy buffers to output place holder
        start_idx = (chunk_idx-1) * N + 1
        end_idx = min(input_size, N*chunk_idx)
        @views copyto!(a_rlwe_in_unpacked[:, start_idx:end_idx],a_out[:,1:end_idx-start_idx+1])
        @views copyto!(b_rlwe_in_unpacked[:, start_idx:end_idx],b_out[:,1:end_idx-start_idx+1])
    end
end

function rlwe_conv(
                a_rlwe_in_unpacked::CuArray{Float64}, 
                b_rlwe_in_unpacked::CuArray{Float64}, 
                kernel::CuArray{Float64, 4}, 
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

function pack_kernel!(
    a_packing_in, b_packing_in,
    a_rlwe_out_unpacked, b_rlwe_out_unpacked,
    pack_id::Int, batch_size::Int, chunk_size::Int,
    M::Int, output_size::Int
)
    i = threadIdx().x
    j = blockIdx().x

    # Only if in bounds
    if i <= batch_size && j <= chunk_size
        out_ind = (pack_id - 1) * batch_size * chunk_size + (i - 1) * chunk_size + j
        if out_ind <= output_size
            for m in 1:M
                a_packing_in[m, i, j] = a_rlwe_out_unpacked[m, out_ind]
                b_packing_in[m, i, j] = b_rlwe_out_unpacked[m, out_ind]
            end
        end
    end
    return
end

function gpu_pack_output!(
                    a_rlwe_out::CuArray{Float64}, b_rlwe_out::CuArray{Float64}, 
                    a_rlwe_out_unpacked::CuArray{Float64}, b_rlwe_out_unpacked::CuArray{Float64},
                    output_size::Int, # (H_out, W_out, C_out, B_outs)
                    shuffle::CuArray{Int32};
                    extract_level=1::Int
                    )
    
    # Reshape A_RLWE_conv
    a_rlwe_out_unpacked = reshape(a_rlwe_out_unpacked, M, output_size)
    b_rlwe_out_unpacked = reshape(b_rlwe_out_unpacked, M, output_size)


    # shuffle
    #a_rlwe_in_unpacked_unshuffle = similar(a_rlwe_out_unpacked)
    #b_rlwe_in_unpacked_unshuffle = similar(b_rlwe_out_unpacked)
    #gpu_column_permute!(a_rlwe_in_unpacked_unshuffle, a_rlwe_out_unpacked, shuffle)
    #gpu_column_permute!(b_rlwe_in_unpacked_unshuffle, b_rlwe_out_unpacked, shuffle)

    #######################
    ### Perform packing ###
    #######################
    number_of_GPU_packing_batches = output_size ÷ (batch_size * (t^β * N)÷M ) + 1


    num_chunks = (t^β * N) ÷ M
    a_packing_in = CUDA.zeros(Float64, M, batch_size, num_chunks)
    b_packing_in = CUDA.zeros(Float64, M, batch_size, num_chunks)

    for pack_id in 1:number_of_GPU_packing_batches
        # Copy A_RLWE_conv to the packing buffers
        @cuda threads=batch_size blocks=num_chunks pack_kernel!(
            a_packing_in, b_packing_in,
            a_rlwe_out_unpacked, b_rlwe_out_unpacked,
            pack_id, batch_size, num_chunks,
            M, output_size
        )

        a_rlwe_out[:,:,pack_id], b_rlwe_out[:,:,pack_id] = gpu_optimal_fast_packing(a_packing_in, b_packing_in, β, array_FFT_ksk_s_d_to_s, g, D,l;start_trace=extract_level)
    end

    return nothing
end


function kernel_column_permute!(
    A_out::CuDeviceMatrix{Float64},  # or Float64 or ComplexF32 etc.
    A_in::CuDeviceMatrix{Float64},
    perm::CuDeviceVector{Int32},
    M::Int, B::Int
)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if row <= M && col <= B
        src_col = perm[col]
        A_out[row, col] = A_in[row, src_col]
    end
    return
end
function gpu_column_permute!(
    A_out::CuArray{T,2}, A_in::CuArray{T,2}, perm::CuArray{Int32}
) where T
    M, B = size(A_in)
    threads = (32, 8)
    blocks = (cld(B, threads[1]), cld(M, threads[2]))
    @cuda threads=threads blocks=blocks kernel_column_permute!(A_out, A_in, perm, M, B)
end
function gpu_server_forward_conv_layer(
        a_rlwe_in::Array{Float64}, 
        b_rlwe_in::Array{Float64},
        a_rlwe_residual::Ref, # Buffers for residual connections
        b_rlwe_residual::Ref, # Buffers for residual connections
        input_shape::Tuple{Int, Int, Int, Int},
        kernel::CuArray{Float64, 4},
        stride::Int,
        padding::Int,
        unshuffle::CuArray{Int32},
        shuffle::CuArray{Int32};
        extract_level=1::Int, # 1 for fast_partial_trace_0, 2 for fast_partial_trace_1_comp_0
        tag=nothing::Union{String, Nothing}, # Tag of layer operation
        projection_kernel=nothing::Union{CuArray{Float64, 4}, Nothing}
        )
    
    H, W, C_in, B = input_shape
    KH, KW, C_in_k, C_out = size(kernel)
    @assert C_in == C_in_k "Input/output channels must match"

    input_size = prod(input_shape)
    
    t_extract = CUDA.@elapsed begin
    # Initialize Input RLWEs
    a_rlwe_in_unpacked = CuArray{Float64}(undef, M, input_size)
    b_rlwe_in_unpacked = CuArray{Float64}(undef, M, input_size)

    # Perform Partial Trace Extract
    gpu_partial_extract_trace!(a_rlwe_in_unpacked, b_rlwe_in_unpacked,
                           CuArray(a_rlwe_in), CuArray(b_rlwe_in), 
                           input_size,  
                           extract_level=extract_level
                        )


    # unshuffle
    #a_rlwe_in_unpacked_unshuffle = similar(a_rlwe_in_unpacked)
    #b_rlwe_in_unpacked_unshuffle = similar(b_rlwe_in_unpacked)
    #gpu_column_permute!(a_rlwe_in_unpacked_unshuffle, a_rlwe_in_unpacked, unshuffle)
    #gpu_column_permute!(b_rlwe_in_unpacked_unshuffle, b_rlwe_in_unpacked, unshuffle)

    a_rlwe_in_unpacked = reshape(a_rlwe_in_unpacked, (M* B, H,W,C_in))
    b_rlwe_in_unpacked = reshape(b_rlwe_in_unpacked, (M* B, H,W,C_in))
    end

    # Save the unpacked RLWEs for residual connections
    t_conv =  CUDA.@elapsed begin
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
    t_pack =  CUDA.@elapsed begin
    BM, H_out, W_out, C_out = size(a_rlwe_out_unpacked)

    output_size =  BM * H_out * W_out * C_out ÷ M
    output_shape = (H_out, W_out, C_out, BM)

    # Compute number of packing batches
    number_of_packing_batches = ceil(Int, output_size / ((t^β * N)÷M ))
    number_of_GPU_packing_batches = ceil(Int, output_size / (batch_size * (t^β * N)÷M ))

    # Preallocate output
    a_rlwe_out = CuArray{Float64}(undef, M, batch_size, number_of_GPU_packing_batches)
    b_rlwe_out = CuArray{Float64}(undef, M, batch_size, number_of_GPU_packing_batches)

    # Pack
    gpu_pack_output!(a_rlwe_out, b_rlwe_out, a_rlwe_out_unpacked, b_rlwe_out_unpacked, output_size, shuffle, extract_level=extract_level)
    #a_rlwe_out_unpacked = reshape(a_rlwe_out_unpacked, M, output_size)
    #b_rlwe_out_unpacked = reshape(b_rlwe_out_unpacked, M, output_size)
    
    #a_rlwe_out, b_rlwe_out = gpu_layer_optimal_fast_packing(
    #    a_rlwe_out_unpacked, 
    #    b_rlwe_out_unpacked, 
    #    β, array_FFT_ksk_s_d_to_s, 
    #    g ,D, l; start_trace=extract_level)
    #a_rlwe_out = reshape(a_rlwe_out, M, batch_size*number_of_GPU_packing_batches)
    #b_rlwe_out = reshape(a_rlwe_out, M, batch_size*number_of_GPU_packing_batches)    
    end
    println("t_extract : ", t_extract, "  t_conv : ", t_conv, "  t_pacl : ", t_pack)
    # Return the output RLWEs
    return a_rlwe_out, b_rlwe_out, t_extract, t_conv, t_pack
end

function gpu_server_forward_fc_layer(
        a_rlwe_in::Array{Float64}, 
        b_rlwe_in::Array{Float64},
        weights::CuArray{Float64, 2}, # Weights for fully connected layer (Input_dim, Output_dim)
        unshuffle::CuArray{Int32}, shuffle::CuArray{Int32};
        extract_level=1::Int, # 1 for fast_partial_trace_0, 2 for fast_partial_trace_1_comp_0
    )
    t_extract = @elapsed begin
    # Implementation similar to server_forward_conv_layer but adapted for fully connected layers
    input_size, output_size = size(weights)

    a_rlwe_in_unpacked = CuArray{Float64}(undef, M, input_size)
    b_rlwe_in_unpacked = CuArray{Float64}(undef, M, input_size)

    # Perform Partial Trace Extract
    gpu_partial_extract_trace!(a_rlwe_in_unpacked, b_rlwe_in_unpacked,
                           CuArray(a_rlwe_in), CuArray(b_rlwe_in), input_size, 
                           extract_level=extract_level
                        )
    
    #a_rlwe_in_unpacked_unshuffle = similar(a_rlwe_in_unpacked)
    #b_rlwe_in_unpacked_unshuffle = similar(b_rlwe_in_unpacked)
    #gpu_column_permute!(a_rlwe_in_unpacked_unshuffle, a_rlwe_in_unpacked, unshuffle)
    #gpu_column_permute!(b_rlwe_in_unpacked_unshuffle, b_rlwe_in_unpacked, unshuffle)
    end
    t_conv = @elapsed begin
    # Perform matrix multiplication
    a_rlwe_out_unpacked = a_rlwe_in_unpacked * weights
    b_rlwe_out_unpacked = b_rlwe_in_unpacked * weights
    end
    # Compute number of packing batches
    number_of_packing_batches = ceil(Int, output_size / ((t^β * N)÷M ))
    number_of_GPU_packing_batches = ceil(Int, output_size / (batch_size * (t^β * N)÷M ))
    
    # Preallocate output
    t_pack = @elapsed begin
        # Preallocate output
    a_rlwe_out = CuArray{Float64}(undef, M, batch_size, number_of_GPU_packing_batches)
    b_rlwe_out = CuArray{Float64}(undef, M, batch_size, number_of_GPU_packing_batches)


    # Pack
    gpu_pack_output!(a_rlwe_out, b_rlwe_out, a_rlwe_out_unpacked, b_rlwe_out_unpacked, output_size, shuffle, extract_level=extract_level)
    end
    return a_rlwe_out, b_rlwe_out, t_extract, t_conv, t_pack
end