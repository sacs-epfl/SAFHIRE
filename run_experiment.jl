#!/usr/bin/env julia
#
# SAFHIRE Experiment Runner
#
# Usage:
#   julia --threads=N run_experiment.jl --model resnet20 --precision 16 --extraction-level 2 [--iterations 2]
#
# Arguments:
#   --model             Model name: resnet20, resnet18, resnet34
#   --precision         Precision in bits: 8, 12, 16
#   --extraction-level  Partial trace extraction level (gamma): 0, 1, 2
#   --iterations        Number of inference iterations (default: 2)
#

###########################
### Argument Parsing    ###
###########################

function parse_args(args)
    parsed = Dict{String, String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i]
            if i + 1 <= length(args) && !startswith(args[i+1], "--")
                parsed[key] = args[i+1]
                i += 2
            else
                error("Missing value for argument: $key")
            end
        else
            error("Unexpected argument: $(args[i])")
        end
    end
    return parsed
end

parsed = parse_args(ARGS)

# Validate required arguments
for required in ["--model", "--precision", "--extraction-level"]
    if !haskey(parsed, required)
        error("Missing required argument: $required\n" *
              "Usage: julia --threads=N run_experiment.jl --model resnet20 --precision 16 --extraction-level 2 [--iterations 2]")
    end
end

model_name = parsed["--model"]
precision = parse(Int, parsed["--precision"])
extraction_level = parse(Int, parsed["--extraction-level"])
iterations = parse(Int, get(parsed, "--iterations", "2"))

# Validate values
@assert model_name in ["resnet20", "resnet18", "resnet34"] "Model must be one of: resnet20, resnet18, resnet34. Got: $model_name"
@assert precision in [8, 12, 16] "Precision must be one of: 8, 12, 16. Got: $precision"
@assert extraction_level in [0, 1, 2] "Extraction level must be one of: 0, 1, 2. Got: $extraction_level"
@assert iterations > 0 "Iterations must be positive. Got: $iterations"

###########################
### Parameter Lookup    ###
###########################
# From paper Table 3: Parameterization for different precision levels and extraction levels

PARAM_TABLE = Dict(
    #        t   α   g   σ_RLWE     γ=0:(D,l)    γ∈[1,2]:(D,l)
    8  => (t=3, α=7, g=2, σ_RLWE=2.0^(-36),
           D=Dict(0 => 512, 1 => 512, 2 => 512),
           l=Dict(0 => 3,   1 => 3,   2 => 3)),
    12 => (t=7, α=4, g=3, σ_RLWE=2.0^(-51),
           D=Dict(0 => 4096, 1 => 65536, 2 => 65536),
           l=Dict(0 => 3,    1 => 2,     2 => 2)),
    16 => (t=5, α=5, g=2, σ_RLWE=2.0^(-51),
           D=Dict(0 => 310, 1 => 310, 2 => 310),
           l=Dict(0 => 5,   1 => 4,   2 => 4)),
)

params = PARAM_TABLE[precision]

# Set _param_* variables before including Constants.jl
_param_p = 2^precision
_param_t = params.t
_param_α = params.α
_param_g = params.g
_param_σ_RLWE = params.σ_RLWE
_param_D = params.D[extraction_level]
_param_l = params.l[extraction_level]

###########################
### Load SAFHIRE        ###
###########################

using FFTW, Primes
using CSV, DataFrames, LinearAlgebra
using Tullio, LoopVectorization

include("src/Types.jl")
include("src/Constants.jl")
include("src/utils/BaseFunctions.jl")
include("src/utils/Operations.jl")
include("src/utils/Encryption.jl")
include("src/utils/KeySwitches.jl")
include("src/PartialTraces.jl")
include("src/Packing.jl")
include("src/Server.jl")
include("src/ServerMultiThreaded.jl")
include("src/Client.jl")

# Include only the needed model config
if model_name == "resnet20"
    include("config/resnet20_cifar_layers.jl")
    model = resnet20_layers_cifar
    dataset_name = "CIFAR10"
    num_classes = 10
elseif model_name == "resnet18"
    include("config/resnet18_tiny_layers.jl")
    model = resnet18_layers_tiny
    dataset_name = "TinyImageNet"
    num_classes = 200
elseif model_name == "resnet34"
    include("config/resnet34_layers.jl")
    model = resnet34_layers
    dataset_name = "ImageNet"
    num_classes = 1000
end

include("examples/utils/example_helpers.jl")
include("examples/utils/examples_multi_thread_inference.jl")

###########################
### Print Configuration ###
###########################

println("=" ^ 60)
println("SAFHIRE Experiment Runner")
println("=" ^ 60)
println("Model:            $(model_name) ($(dataset_name))")
println("Precision:        $(precision) bits (p = 2^$(precision))")
println("Extraction Level: $(extraction_level)")
println("Iterations:       $(iterations)")
println("Threads:          $(Threads.nthreads())")
println("FHE Parameters:   t=$(t), α=$(α), M=$(M), g=$(g)")
println("                  D=$(D), l=$(l), β=$(β)")
println("                  σ_RLWE=$(σ_RLWE)")
println("=" ^ 60)

###########################
### Run Experiment      ###
###########################

# Configure environment
BLAS.set_num_threads(1)

# Define weight space and initialize model
weight_space = [i/1.0 for i=-3:4]
log_p = round(log2(p))

kernel_weights, projection_weights, fc_weights, biases = initialize_model_inference!(model, weight_space)

# Initialize output logger
global output_logger = DataFrame(
    real_output = Array{Int64}[],
    fhe_output = Array{Int64}[],
    max_err = Int64[]
)
global nb_threads = Threads.nthreads()

# Create logs directory
log_dir = "./logs"
mkpath(log_dir)
base_name = "$(model_name)_$(dataset_name)_P$(precision)_Tr$(extraction_level)_Thread$(nb_threads)"

for iter in 1:iterations
    # Pick random input data
    global nch_input_data_vec = vec(rand([i for i=-15:16], model[1].input_shape) / p)

    clear_output_data = clear_model_inference(
        copy(nch_input_data_vec),
        model,
        kernel_weights,
        projection_weights,
        fc_weights,
        biases
    )

    inference_time = @elapsed begin
    fhe_output_data, log_data = run_multithreaded_hybrid_inference(
        model,
        copy(nch_input_data_vec),
        kernel_weights,
        projection_weights,
        fc_weights,
        biases,
        extraction_level
    )
    end

    max_error = maximum(abs.(round.(Int64, p*fhe_output_data) .- round.(Int64, vec(clear_output_data) * p)))
    println("Iteration $iter: Max error = $max_error, Inference time = $(inference_time)s")
    @assert max_error == 0 "Max error is not zero! Got $max_error"

    # Log outputs
    push!(output_logger, (
        real_output = round.(Int64, vec(clear_output_data) * p),
        fhe_output = round.(Int64, p * [fhe_output_data[i] for i=1:num_classes]),
        max_err = max_error
    ))

    CSV.write("$(log_dir)/$(base_name)_run_$(iter).csv", log_data)
    CSV.write("$(log_dir)/$(base_name)_output.csv", output_logger)
end

println("=" ^ 60)
println("Experiment completed successfully.")
println("Logs saved to: $(log_dir)/$(base_name)_*.csv")
println("=" ^ 60)
