function partial_trace_0!(a_tilded::Vector{Float64},
                        b_tilded::Vector{Float64},
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        g::Int, D::Int, l::Int,
                        # scratch buffers passed in
                        a_sum::Vector{Float64},
                        b_sum::Vector{Float64},
                        a_d::Vector{Float64},
                        b_d::Vector{Float64}
                        )
    # In-place partial trace using in-place key-switch with external buffers
    π = t - 1
    aux = 2
    while π > 1
        while π % aux != 0
            aux = nextprime(aux + 1)
        end
        π = π ÷ aux

        # initialize sums to current a_tilded/b_tilded
        copyto!(a_sum, a_tilded)
        copyto!(b_sum, b_tilded)

        # accumulate for k = 1:(aux-1)
        for k=1:(aux-1)
            d = powermod(g, k*π , M)
            #
            inplace_power(a_d, a_tilded, d)
            inplace_power(b_d, b_tilded, d)
            #
            rlwe_key_switch_accumulate!(a_sum,b_sum,a_d, b_d, array_FFT_ksk_s_prime_to_s[d], D, l)
        end

        # Write sum back into tilded arrays
        copyto!(a_tilded, a_sum)
        copyto!(b_tilded, b_sum)
    end
    return nothing
end

"""
    partial_trace(C::RLWE, j::Int64,
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        D::Int64,l::Int64
                        )

    Compute Homomorphically an RLWE encryption of Tr_j(μ) given an RLWE encryption of μ and the Key Switching keys from s(X^d) to s(X)
"""
function partial_trace(C::RLWE, j::Int64,
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        D::Int64,l::Int64
                        )
    #
    #   Compute a RLWE encruption of Tr_j(μ) given a RLWE encryption
    #   of μ and the Key Switching keys from s(X^d) to s(X)
    #
    a_sum = copy(C[1])
    b_sum = copy(C[2])
    #
    for k=1:(t-1)
        d = k*t^j + 1
        #
        a_d, b_d = rlwe_key_switch((power(C[1], d), power(C[2],d)),array_FFT_ksk_s_prime_to_s[d], D, l)
        a_sum .+= a_d
        b_sum .+= b_d
    end
    #
    return (a_sum, b_sum)
end

function partial_trace!(a_tilded::Vector{Float64},
                        b_tilded::Vector{Float64}, j::Int64,
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        D::Int, l::Int,
                        # scratch buffers passed in
                        a_sum::Vector{Float64},
                        b_sum::Vector{Float64},
                        a_d::Vector{Float64},
                        b_d::Vector{Float64}
                        )
    # In-place partial trace using in-place key-switch with external buffers

    # Initialize sums to current a_tilded/b_tilded
    copyto!(a_sum, a_tilded)
    copyto!(b_sum, b_tilded)

    # Accumulate for k = 1:(t-1)
    for k=1:(t-1)
        d = k*t^j + 1
        #
        inplace_power(a_d, a_tilded, d)
        inplace_power(b_d, b_tilded, d)
        #
        rlwe_key_switch_accumulate!(a_sum,b_sum,a_d, b_d, array_FFT_ksk_s_prime_to_s[d], D, l)
    end 

    # Write sum back into tilded arrays
    copyto!(a_tilded, a_sum)
    copyto!(b_tilded, b_sum)
    return nothing
end

function fast_partial_extract!(a_out::Vector{Float64}, b_out::Vector{Float64},
                            a_RLWE::Array{Float64}, b_RLWE::Array{Float64},
                            i::Int64
                            )
    # Fast partial trace extract for level 0
    fill!(a_out, 0.0)
    fill!(b_out, 0.0)

    @views multiply_pol_Ω_star_i_pow_minus_j!(a_out, a_RLWE[:,1], i, 1)
    @views multiply_pol_Ω_star_i_pow_minus_j!(b_out, b_RLWE[:,1], i, 1)

    return
end

function fast_partial_trace_0!(a_out::Vector{Float64}, b_out::Vector{Float64},
                            a_RLWE::Array{Float64}, b_RLWE::Array{Float64},
                            i::Int64
                            )
    fill!(a_out, 0.0)
    fill!(b_out, 0.0)
    for j=1:(t-1)
        @views multiply_pol_Ω_star_i_pow_minus_j!(a_out, a_RLWE[:,j], i, j)
        @views multiply_pol_Ω_star_i_pow_minus_j!(b_out, b_RLWE[:,j], i, j)
    end

    return
end
function fast_partial_trace_1_comp_0!(a_out::Vector{Float64}, b_out::Vector{Float64},
                            a_RLWE::Array{Float64}, b_RLWE::Array{Float64},
                            i::Int64
                            )
    # To make memory more efficient it is assumed that a_RLWE[:,j], b_RLWE[:,j]
    # Encrypts μ(X^i(k*t+1)) pour i ∈ [1;t-1] ; k ∈ [0;t-1]
    fill!(a_out, 0.0)
    fill!(b_out, 0.0)
    for j=1:(t-1)
        for k=0:(t-1)
            @views multiply_pol_Ω_star_i_pow_minus_j!(a_out, a_RLWE[:,j+k*(t-1)], i, j*(k*t+1))
            @views multiply_pol_Ω_star_i_pow_minus_j!(b_out, b_RLWE[:,j+k*(t-1)], i, j*(k*t+1))
        end
    end

    return
end
