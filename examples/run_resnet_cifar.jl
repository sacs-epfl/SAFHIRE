using FFTW
using CSV, DataFrames

include("../src/Types.jl")
include("../src/Constants.jl")
include("../src/utils/BaseFunctions.jl")
include("../src/utils/Operations.jl")
include("../src/utils/Encryption.jl")
include("../src/utils/KeySwitches.jl")
include("../src/PartialTraces.jl")
include("../src/Packing.jl")
include("../src/Server.jl")
include("../src/Client.jl")

include("../config/resnet20_cifar_layers.jl")
include("./utils/example_helpers.jl")
include("./utils/examples_hybrid_inference.jl")

## Load model configuration
model = resnet20_layers_cifar

## Define Constants
weight_space = [i/1.0 for i=-3:4]
log_p = round(log2(p)) # Precision

# Initialize kernel weights for all layers
kernel_weights, projection_weights, fc_weights = initialize_model_inference!(model, weight_space)

# Initialize Logger for outputs
global output_logger = DataFrame(
    real_output = Array{Int64}[],
    fhe_output = Array{Int64}[],
    max_err = Int64[]
)

for iter in 1:2
    # Pick random input data
    global nch_input_data_vec = vec(rand([i for i=-15:16], model[1].input_shape) / p)

    clear_output_data = clear_model_inference(
        copy(nch_input_data_vec), 
        model, 
        kernel_weights, 
        projection_weights, 
        fc_weights
    )
    #println("Clear output data vector: ", round.(Int64,vec(clear_output_data)[1:10] *p))

    inference_time = @elapsed begin
    fhe_output_data, log_data = run_hybrid_inference(
        model, 
        copy(nch_input_data_vec), 
        kernel_weights, 
        projection_weights, 
        fc_weights,
        0 # extract_level
    )
    end 

    #println("Clear output data vector: ", round.(Int64,vec(clear_output_data)[1:10] *p))
    #println("Final output data vector: ", round.(Int64,p*[fhe_output_data[i] for i=1:10]) )
    println("Max error: ", maximum(abs.(round.(Int64,p*fhe_output_data) - round.(Int64,vec(clear_output_data) *p))))
    println("Inference time: ", inference_time, " seconds")

    # Push ouput to DataFrame
    push!(output_logger, (
        real_output = round.(Int64,vec(clear_output_data) *p),
        fhe_output = round.(Int64,p*[fhe_output_data[i] for i=1:10]),
        max_err = maximum(abs.(round.(Int64,p*fhe_output_data) - round.(Int64,vec(clear_output_data) *p)))
    ))

    CSV.write("./logs/ResNet20_Cifar10_M$(M)_P$(log_p)_run_$iter.csv", log_data)
    
    CSV.write("./logs/ResNet20_Cifar10_M$(M)_P$(log_p)_output.csv", output_logger)
end