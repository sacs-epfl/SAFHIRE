function load_gpu_params(
    model,
    kernel_weights, 
    projection_weights, 
    fc_weights
)
    gpu_kernel_weights = CuArray{Float64, 4}[]
    gpu_projection_weights = Union{CuArray{Float64, 4}, Nothing}[]
    gpu_fc_weights = CuArray(fc_weights)
    for layer in model
        if !(layer.tag == "fc")
            # For convolutional layers, initialize kernel weights
            push!(gpu_kernel_weights, CuArray(kernel_weights[layer.id]))
            if layer.tag == "residual_projection"
                push!(gpu_projection_weights, CuArray(projection_weights[layer.id]))
            else
                push!(gpu_projection_weights, nothing)
            end
        end
    end
    return gpu_kernel_weights, gpu_projection_weights, gpu_fc_weights
end

function gpu_run_hybrid_inference(
    model, 
    input_data, 
    kernel_weights, 
    projection_weights, 
    fc_weights, 
    extract_level
    )

    # Initialize log for timings and sizes
    log = DataFrame(
    Iteration = Int[],
    t_encr = Float64[],
    t_decrpt = Float64[],
    t_relu = Float64[],
    t_server = Float64[],
    t_extract = Float64[],
    t_conv = Float64[],
    t_pack = Float64[],
    rlwe_size_up = Float64[],
    rlwe_size_down = Float64[]
    )
    
    global unshuffle = nothing

    # Initialize residual connection
    global a_rlwe_residual = Ref{Union{Nothing, CuArray{Float64}}}(nothing) 
    global b_rlwe_residual = Ref{Union{Nothing, CuArray{Float64}}}(nothing) 

    global nch_input_data_vec = copy(input_data)
    for layer in model

        if layer.id == 1
            temp = Int32.(collect(RandomPermutation(prod(layer.input_shape)))) # Random permutation for first layer
            global unshuffle = CuArray(temp)
        end
        nch_input_shape = layer.input_shape # (H, W, C_in, B)
        stride = layer.stride
        padding = layer.padding

        if layer.tag == "fc"
            println("Processing layer: ", layer.id, " FC ")
        else
            println("Processing layer: ", layer.id, " Kernel shape: ", layer.kernel_shape, " Stride: ", stride, " Padding: ", padding)
        end
        # Encrypt input data to RLWE
        time_enc = @elapsed begin
        a_rlwe_in, b_rlwe_in = encrypt_pows_input_vec_to_rlwe(
            nch_input_data_vec, 
            prod(nch_input_shape), 
            FFT_Key_RLWE, σ_RLWE; 
            trace_level=extract_level
        )
        end
        # Send data to server
        RLWE_upload_size = (Base.summarysize(a_rlwe_in) + Base.summarysize(b_rlwe_in)) / (1024 * 1024)

        # Get ouptut size
        if layer.tag == "fc"
            output_size = prod(layer.output_shape)
        else
            _, output_size = get_conv_output_size(nch_input_shape, layer.kernel_shape, stride, padding)
        end

        time_server = @elapsed begin
        shuffle = Int32.(collect(RandomPermutation(output_size)))
        GPUshuffle = CuArray(shuffle)

        # Perform server forward layer operation
        @time begin
        if layer.tag == "fc"
            temp = Int32.(collect(Permutation(output_size)))
            GPUshuffle = CuArray(temp)
            # For fully connected layers
            a_rlwe_out, b_rlwe_out, t_extract, t_conv, t_pack = gpu_server_forward_fc_layer(
                a_rlwe_in, b_rlwe_in,
                fc_weights,
                unshuffle,
                GPUshuffle;
                extract_level=extract_level
            )
        else
            # For convolutional layers and average pooling layers
            a_rlwe_out, b_rlwe_out, t_extract, t_conv, t_pack = gpu_server_forward_conv_layer(
                a_rlwe_in, b_rlwe_in,
                a_rlwe_residual, b_rlwe_residual,
                nch_input_shape,
                kernel_weights[layer.id], 
                stride, padding,
                unshuffle,
                GPUshuffle;
                extract_level=extract_level,
                tag=layer.tag,
                projection_kernel=projection_weights[layer.id]
            )
        end
        end
        end

        # Store inverse of shuffle for next layer
        shuffleinv = Int32.(collect(inv(Permutation(shuffle))))
        global unshuffle = CuArray(shuffleinv)

        # Receive data from server
        RLWE_download_size = (Base.summarysize(a_rlwe_out) + Base.summarysize(b_rlwe_out)) / (1024 * 1024)

        time_dcrpt = @elapsed begin

        ## Put data on CPU
        a_rlwe_out = Array(a_rlwe_out)
        b_rlwe_out = Array(b_rlwe_out)

        # Decrypt output data
        output_data_vec = decrypt_output(
            a_rlwe_out, b_rlwe_out, 
            output_size, 
            FFT_Key_RLWE, β
        )
        end

        # Process output data (e.g., reshape, apply activation, etc.)

        # Apply ReLU activation function
        time_relu = @elapsed begin
        if !(layer.tag == "avg_pool" || layer.tag == "fc")
            relu!(output_data_vec) # Apply ReLU activation function
        end
        end

        # Update nch_input_data_vec for the next layer
        global nch_input_data_vec = output_data_vec
        push!(log, (
            Iteration = layer.id,
            t_encr = time_enc,
            t_decrpt = time_dcrpt,
            t_relu = time_relu,
            t_server = time_server,
            t_extract = t_extract,
            t_conv = t_conv,
            t_pack = t_pack,
            rlwe_size_up = RLWE_upload_size,
            rlwe_size_down = RLWE_download_size
        ))
    end

    # Return the log of timings and sizes
    return nch_input_data_vec, log
end