function get_max_layer_size(model)
    max_input_size = 0
    for layer in model
        if prod(layer.input_shape) > max_input_size
            max_input_size = prod(layer.input_shape)
        end
    end
    return max_input_size
end

function initialize_model_inference!(
    model,
    weight_space
)
    kernel_weights = Array{Float64, 4}[]
    projection_weights = Union{Array{Float64,4}, Nothing}[]
    fc_weights = Array{Float64}
    # Initialize kernel weights for all layers
    for layer in model
        if !(layer.tag == "fc")
            # For convolutional layers, initialize kernel weights
            push!(kernel_weights, rand(weight_space, layer.kernel_shape))
            if layer.tag == "residual_projection"
                W,H,C_in,C_out = layer.kernel_shape

                push!(projection_weights, rand(weight_space, (1, 1, C_in, C_out)))
            else
                push!(projection_weights, nothing)
            end
        end
        if layer.tag == "fc"
            # For fully connected layers, initialize weights
            input_dim = prod(layer.input_shape)
            output_dim = prod(layer.output_shape)
            #
            fc_weights = rand(weight_space, input_dim, output_dim)
        end
    end

    return kernel_weights, projection_weights, fc_weights
end

function clear_model_inference(
    nch_input_data_vec, 
    resnet20_layers_cifar, 
    kernel_weights, 
    projection_weights, 
    fc_weights
)
    clear_input_data_vec = copy(nch_input_data_vec)
    clear_input_data = reshape(clear_input_data_vec, resnet20_layers_cifar[1].input_shape)

    # Initialize residual connection
    clear_residual = Ref(zeros(Float64, resnet20_layers_cifar[1].input_shape))
    
    # Print initial input data
    for layer in resnet20_layers_cifar
        nch_input_shape = layer.input_shape # (H, W, C_in, B)
        stride = layer.stride
        padding = layer.padding

        # Perform convolution
        if layer.tag == "fc"
            input_size = prod(layer.input_shape)
            clear_input_data = reshape(clear_input_data, (1, input_size))
            clear_output_data = clear_input_data * fc_weights
        elseif layer.tag == "avg_pool"
            # For average pooling layers
            clear_output_data = sum(clear_input_data, dims=(1, 2)) # Sum over H and W dimensions
        else
            # For convolutional layers
            clear_output_data  = conv(clear_input_data, kernel_weights[layer.id]; stride=(stride, stride), pad=(padding, padding))
        
            if layer.tag == "residual_in"
                clear_residual[] = clear_input_data
            elseif layer.tag == "residual_projection"
                # Perform projection
                clear_residual[] = conv(clear_input_data, projection_weights[layer.id]; stride = (2,2), pad = (0,0))
            elseif layer.tag == "residual_add"
                # Add residual connection if needed
                clear_output_data .+= clear_residual[]
            end
        end
        
        # Project to Discrete Torus to get read of rounding encrypt_poly_star_to_rlwe_star
        clear_output_data = project_to_discrete_torus.(clear_output_data, p)
        # Requantize the output data
        #requantize_output!(clear_output_data, scaling)

        if !(layer.tag == "avg_pool" || layer.tag == "fc")
            # Apply ReLU activation function
            clear_output_data = max.(clear_output_data, 0.0) # Apply ReLU activation function
        end
        if layer.tag == "avg_pool"
            #clear_output_data ./= layer.input_shape[1] * layer.input_shape[2]
            clear_output_data = project_to_discrete_torus.(clear_output_data, p)
        end

        clear_input_data = clear_output_data
    end
    return clear_input_data
end