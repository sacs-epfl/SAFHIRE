
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