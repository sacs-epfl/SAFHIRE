function rotate_poly!(a::AbstractVector{Float64},Π::AbstractVector{Float64},ind::Int64)
    #
    @inbounds for i = 1:M
        a[i] += Π[mod(i-ind,M)+1]
    end
end

function rotate_poly_sub!(a::Vector{Float64},Π::AbstractVector{Float64},ind::Int64)
    #
    @inbounds @simd for i = 1:M
        a[i] -= Π[mod(i-ind,M)+1]
    end
end

function inplace_power(buffer::Vector{Float64}, Π::Vector{Float64},d::Int64)
    fill!(buffer, 0.0)
    @inbounds for i = 1:M
        buffer[mod((i-1)*d,M)+1] += Π[i]
    end
end

function batch_inplace_power(buffer::AbstractArray{Float64}, Π::AbstractArray{Float64},d::Int64)
    b_size = size(buffer, 2)
    fill!(buffer, 0.0)
    Threads.@threads for b in 1:b_size
        @inbounds for i = 1:M
            buffer[mod((i-1)*d,M)+1,b] += Π[i,b]
        end
    end
end

function power(Π::AbstractArray{Float64}, d::Int64)
    #
    #   Calculates the coefficients of the polynomial Π(X^d)
    #
    Π_X_d = zeros(M)
    for i = 1:M
        Π_X_d[mod((i-1)*d,M)+1] += Π[i]
    end
    #
    return Π_X_d
end

function multiply_pol_Ω_star_i_pow_minus_j!(Π_out::Vector{Float64},Π::AbstractVector{Float64},i::Int64,j::Int64)
    idx1 = mod(-i*j, M)
    idx2 = mod(- j * (N + i%M_over_t) , M)

    rotate_poly!(Π_out, Π, idx1+1)
    rotate_poly_sub!(Π_out, Π, idx2+1)
end