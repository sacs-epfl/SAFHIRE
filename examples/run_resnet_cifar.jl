using FFTW

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

# Initialize kernel weights for all layers
kernel_weights, projection_weights, fc_weights = initialize_model_inference!(model, weight_space)

# Pick random input data
global nch_input_data_vec = vec(rand([i for i=-15:16], model[20].input_shape) / p)
global clear_input_data_vec = copy(nch_input_data_vec)

clear_input_data = clear_model_inference(
    nch_input_data_vec, 
    model[20:21], 
    kernel_weights, 
    projection_weights, 
    fc_weights
)
println("Clear output data vector: ", round.(Int64,vec(clear_input_data)[1:10] *p))

using CSV, DataFrames

inference_time = @elapsed begin
log_data = run_hybrid_inference(
    model[20:21], 
    nch_input_data_vec, 
    kernel_weights, 
    projection_weights, 
    fc_weights,
    2 # extract_level
)
end 

println("Clear output data vector: ", round.(Int64,vec(clear_input_data)[1:10] *p))
println("Final output data vector: ", round.(Int64,p*[nch_input_data_vec[i] for i=1:10]) )
println("Inference time: ", inference_time, " seconds")

log_p = round(log2(p))
CSV.write("./logs/ResNet20_Cifar10_M$(M)_P$(log_p).csv", log_data)