### Compute automophism indices for keyswitching ###
indices = compute_automorphism_indices(g)
#
const array_ksk_s_d_to_s = Array{Vector{RLWE}}(undef, M)
for d in indices
    array_ksk_s_d_to_s[d] = compute_rlwe_keyswitching_key(FFT_Key_RLWE, convert.(Int64,(power(convert.(Float64, Key_RLWE),d))),σ_RLWE,D,l)
end
array_ksk_s_d_to_s[M-1] = compute_rlwe_keyswitching_key(FFT_Key_RLWE, convert.(Int64,(power(convert.(Float64, Key_RLWE),M-1))),σ_RLWE,D,l)
const array_FFT_ksk_s_prime_to_s = Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}}(undef, M)
for d in indices
    array_FFT_ksk_s_prime_to_s[d] = Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}(undef, l)
    for i=1:l
        array_FFT_ksk_s_prime_to_s[d][i] = (op_fft_M*array_ksk_s_d_to_s[d][i][1], op_fft_M*array_ksk_s_d_to_s[d][i][2])
    end
end

const FFT_automorphosim_ksk = Array{ComplexF64}(undef, M, l, 2 ,M)
for d in indices
    for i=1:l
        Threads.@threads for m=1:M
            FFT_automorphosim_ksk[m,i,1,d] = array_FFT_ksk_s_prime_to_s[d][i][1][m]
            FFT_automorphosim_ksk[m,i,2,d] = array_FFT_ksk_s_prime_to_s[d][i][2][m]
        end
    end
end

function encrypt_pows_input_vec_to_rlwe(nch_input_data_vec::Vector{Float64}, 
                                        input_size::Int64, 
                                        FFT_Key_RLWE::FFT_Key, 
                                        σ::Float64;
                                        trace_level=1::Int64
                                        )
    c = input_size ÷ N
    if trace_level==0
        a_rlwe_in = Array{Float64}(undef, M, 1, c+1)
        b_rlwe_in = Array{Float64}(undef, M, 1, c+1)
        for i in 1:(c+1)
            if i == (c+1)
                message = [nch_input_data_vec[(i-1)*N+1:end]; zeros(M- input_size % N) ]
                a_rlwe_in[:, 1, i], b_rlwe_in[:, 1, i] = encrypt_poly_to_rlwe(message, FFT_Key_RLWE, σ)
            else
                message = [nch_input_data_vec[(i-1)*N+1:i*N]; zeros(M-N)]
                a_rlwe_in[:, 1 , i], b_rlwe_in[:, 1, i] = encrypt_poly_to_rlwe(message, FFT_Key_RLWE, σ)
            end
        end
    elseif trace_level==1
        a_rlwe_in = Array{Float64}(undef, M, t-1, c+1)
        b_rlwe_in = Array{Float64}(undef, M, t-1, c+1)
        for i in 1:(c+1)
            for j=1:(t-1)
                if i == (c+1)
                    message = [nch_input_data_vec[(i-1)*N+1:end]; zeros(M- input_size % N) ]
                    a_rlwe_in[:,j, i], b_rlwe_in[:,j, i] = encrypt_poly_to_rlwe(power(message,j), FFT_Key_RLWE, σ)
                else
                    message = [nch_input_data_vec[(i-1)*N+1:i*N]; zeros(M-N)]
                    a_rlwe_in[:,j , i], b_rlwe_in[:,j, i] = encrypt_poly_to_rlwe(power(message,j), FFT_Key_RLWE, σ)
                end
            end
        end
    elseif trace_level==2
        a_rlwe_in = Array{Float64}(undef, M, t*(t-1), c+1)
        b_rlwe_in = Array{Float64}(undef, M, t*(t-1), c+1)
        for i in 1:(c+1)
            for j=1:(t-1)
                for k=0:(t-1)
                    if i == (c+1)
                        message = [nch_input_data_vec[(i-1)*N+1:end]; zeros(M- input_size % N) ]
                        a_rlwe_in[:,k*(t-1)+j, i], b_rlwe_in[:,k*(t-1)+j, i] = encrypt_poly_to_rlwe(power(message,j*(k*t+1)), FFT_Key_RLWE, σ)
                    else
                        message = [nch_input_data_vec[(i-1)*N+1:i*N]; zeros(M-N)]
                        a_rlwe_in[:,k*(t-1)+j , i], b_rlwe_in[:,k*(t-1)+j, i] = encrypt_poly_to_rlwe(power(message,j*(k*t+1)), FFT_Key_RLWE, σ)
                    end
                end
            end
        end
    end
    return a_rlwe_in, b_rlwe_in
end

function decrypt_output(a_out::Array{Float64}, b_out::Array{Float64}, 
                        output_size::Int64, 
                        FFT_Key_RLWE::FFT_Key, 
                        β::Int64;
                        factor=1::Int64)

    #Assign output vector
    output_data_vec = Vector{Float64}(undef, output_size)
    
    # Compute number of packing batches
    number_of_packing_batches = size(a_out)[2]


    for batch_id = 1:number_of_packing_batches
        aux = mod_Φ(decrypt_rlwe_to_poly((a_out[:, batch_id], b_out[:,batch_id]), FFT_Key_RLWE))
        for j=1:((t^β * N)÷M)
            idx = (batch_id - 1) *(t^β * N)÷M + j
            if idx > output_size
                break
            end
            #output_data_vec[idx] = project_to_discrete_torus((M_inv * project_to_discrete_torus(aux[(j-1)*M÷(t^β)+1], p)),p)
            #if factor != 1
            #    output_data_vec[idx] = rounddown(output_data_vec[idx]*p / factor) / p
            #end
            output_data_vec[idx] = project_to_discrete_torus((M_inv * project_to_discrete_torus(aux[(j-1)*M÷(t^β)+1], p)),p)
            if factor != 1
                output_data_vec[idx] = project_to_discrete_torus(output_data_vec[idx] / factor,p)
            end
        end
    end

    return output_data_vec
end

function decrypt_batch_output(a_out::Array{Float64}, b_out::Array{Float64}, 
                        output_size::Int64, 
                        FFT_Key_RLWE::FFT_Key, 
                        β::Int64;
                        factor=1::Int64)

    output_data_vec = zeros(output_size)

    nb_packing_batches = size(a_out)[3]
    for i = 1:nb_packing_batches
        for j=1:batch_size
            poly = mod_Φ(decrypt_rlwe_to_poly((a_out[:,j,i], b_out[:,j,i]), FFT_Key_RLWE))
            for pck_idx =1:(t^β * N ÷ M)
                
                linear_idx = (i-1)*(batch_size * (t^β * N)÷M) + (j-1) * (t^β * N ÷ M)  + pck_idx
                if linear_idx>output_size
                    continue
                end
                output_data_vec[linear_idx] = project_to_discrete_torus(M_inv *project_to_discrete_torus(poly[(pck_idx-1)*M÷(t^β)+1], p),p)
            end
        end
    end

    return output_data_vec
end

function requantize_output!(output_data_vec::Vector{Float64}, scaling::Int64)
    # Requantize the output data vector
    for i in 1:length(output_data_vec)
        output_data_vec[i] = round(output_data_vec[i] * (2^scaling)) / (2^scaling)
    end
    clamp!(output_data_vec, -15, 16)

    output_data_vec ./= p
end

function requantize_output!(output_data_vec::AbstractArray{Float64}, scaling::Int64)
    # Requantize the output data vector

    output_data_vec .= round.(output_data_vec .* (2^scaling)) ./ (2^scaling)

    clamp!(output_data_vec, -15, 16)

    output_data_vec ./= p
end

function relu!(data::Vector{Float64})
    # Apply ReLU activation function
    data .= max.(real_to_torus.(data), 0.0)
end