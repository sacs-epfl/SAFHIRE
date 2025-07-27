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
function batch_partial_trace_0!(a_tilded::AbstractArray{Float64},
                        b_tilded::AbstractArray{Float64},
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        g::Int, D::Int, l::Int,
                        # scratch buffers passed in
                        a_sum::Array{Float64},
                        b_sum::Array{Float64},
                        a_d::Array{Float64},
                        b_d::Array{Float64}
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
            batch_inplace_power(a_d, a_tilded, d)
            batch_inplace_power(b_d, b_tilded, d)
            #
            batch_rlwe_key_switch_accumulate!(a_sum,b_sum,a_d, b_d, array_FFT_ksk_s_prime_to_s[d], D, l)
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

function batch_partial_trace!(a_tilded::AbstractArray{Float64},
                        b_tilded::AbstractArray{Float64}, j::Int64,
                        array_FFT_ksk_s_prime_to_s::Array{Vector{Tuple{Array{ComplexF64}, Array{ComplexF64}}}},
                        D::Int, l::Int,
                        # scratch buffers passed in
                        a_sum::Array{Float64},
                        b_sum::Array{Float64},
                        a_d::Array{Float64},
                        b_d::Array{Float64}
                        )
    # In-place partial trace using in-place key-switch with external buffers

    # Initialize sums to current a_tilded/b_tilded
    copyto!(a_sum, a_tilded)
    copyto!(b_sum, b_tilded)

    # Accumulate for k = 1:(t-1)
    for k=1:(t-1)
        d = k*t^j + 1
        #
        batch_inplace_power(a_d, a_tilded, d)
        batch_inplace_power(b_d, b_tilded, d)
        #
        batch_rlwe_key_switch_accumulate!(a_sum,b_sum,a_d, b_d, array_FFT_ksk_s_prime_to_s[d], D, l)
    end 

    # Write sum back into tilded arrays
    copyto!(a_tilded, a_sum)
    copyto!(b_tilded, b_sum)
    return nothing
end

function batch_partial_trace!(a_tilded::AbstractArray{Float64},
                        b_tilded::AbstractArray{Float64}, j::Int64,
                        FFT_automorphosim_ksk::Array{ComplexF64},
                        D::Int, l::Int,
                        # scratch buffers passed in
                        a_sum::Array{Float64},
                        b_sum::Array{Float64},
                        a_d::Array{Float64},
                        b_d::Array{Float64},
                        a_buf::Array{ComplexF64},
                        b_buf::Array{ComplexF64},
                        aux_a::Array{ComplexF64}
                        )
    # In-place partial trace using in-place key-switch with external buffers

    # Initialize sums to current a_tilded/b_tilded
    copyto!(a_sum, a_tilded)
    copyto!(b_sum, b_tilded)

    # Accumulate for k = 1:(t-1)
    fill!(a_buf, 0.0)
    fill!(b_buf, 0.0)
    for k=1:(t-1)
        d = k*t^j + 1
        #
        batch_inplace_power(a_d, a_tilded, d)
        batch_inplace_power(b_d, b_tilded, d)
        #
        #println("Batch size : ", size(a_tilded, 2))
        @views batch_rlwe_key_switch_accumulate!(a_sum,b_sum,a_d, b_d, FFT_automorphosim_ksk[:,:,:,d], D, l,a_buf,b_buf,aux_a)
    end 

    b_size = size(a_buf, 2)

    batch_ifft_plans[b_size]*a_buf
    batch_ifft_plans[b_size]*b_buf

    a_sum .+= real.(a_buf)
    b_sum .+= real.(b_buf)
    

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

function multi_threaded_fast_partial_trace_0!(
    a_out::AbstractArray{Float64}, a_pows_RLWE::AbstractArray{Float64}
)
    N_bis = min(N, size(a_out, 2))
    @inbounds Threads.@threads for idx in 1:N_bis*M
        m = (idx-1) % M + 1
        i = (idx-1) ÷ M + 1
        acc = 0.0
        
        @simd for j in 1:(t - 1)
            idx1 = -(i - 1)* j
            idx2 = -j * (N + mod(i - 1, M_over_t))
            acc += a_pows_RLWE[mod(m - idx1 - 1, M) + 1, j]
            acc -= a_pows_RLWE[mod(m - idx2 - 1, M) + 1, j]
        end
        a_out[m, i] =  acc
    end
    return
end

#idx1_array = Array{Int}(undef, k, j, m, n)

function multi_threaded_fast_partial_trace_1_comp_0!(
    a_out::AbstractArray{Float64}, a_pows_RLWE::AbstractArray{Float64}
)
    N_bis = min(N, size(a_out, 2))
    Threads.@threads for tid in 1:Threads.nthreads()
        chunk = ceil(Int, N_bis * M / Threads.nthreads())
        start_idx = (tid - 1) * chunk + 1
        end_idx = min(tid * chunk, N_bis * M)   

        @inbounds for idx in start_idx:end_idx
            m = (idx-1) % M + 1
            i = (idx-1) ÷ M + 1
            acc = 0.0
            begin
                for k=0:(t-1)
                    for j in 1:(t - 1)
                        pow = j*(k*t+1)
                        idx1 = -(i - 1)* pow
                        idx2 = -pow * (N + mod(i - 1, M_over_t))
                        acc += a_pows_RLWE[mod(m - idx1 - 1, M) + 1, j+k*(t-1)]
                        acc -= a_pows_RLWE[mod(m - idx2 - 1, M) + 1, j+k*(t-1)]
                    end
                end
                
                a_out[m, i] = acc
            end
        end
    end
    return
end

function l2_blocked_trace!(
    a_out::AbstractMatrix{Float64}, a_pows_RLWE::AbstractMatrix{Float64},
    i_block_size::Int = 8
)
    N_bis = min(N, size(a_out, 2))

    #@inbounds Threads.@threads for i_start in 1:i_block_size:N_bis
    Threads.@threads for i_block in 1:i_block_size:N_bis
        i_end = min(i_block + i_block_size - 1, N_bis)
        for i in i_block:i_end
            for k = 0:(t-1)
                for j = 1:(t-1)
                    pow = j * (k*t + 1)
                    col = j + k * (t - 1)
                    idx1 = mod(- (i - 1) * pow, M)
                    idx2 = mod(- pow * (N + mod(i - 1, M_over_t)), M)

                    # Access 1 column at a time
                    @inbounds @simd for m = 1:idx1
                        row1 = M + (m - idx1)
                        a_out[m, i] += a_pows_RLWE[row1, col]
                    end
                    @inbounds @simd for m= (idx1+1):M
                        row1 = m - idx1
                        a_out[m, i] += a_pows_RLWE[row1, col]
                    end
                    @inbounds @simd for m = 1:idx2
                        row2 = M + (m - idx2)
                        a_out[m, i] -= a_pows_RLWE[row2, col]
                    end
                    @inbounds @simd for m= (idx2+1):M
                        row2 = m - idx2
                        a_out[m, i] -= a_pows_RLWE[row2, col]
                    end
                end
            end
        end
    end
end