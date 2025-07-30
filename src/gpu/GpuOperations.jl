using CUDA
CUDA.set_runtime_version!(v"12.6")
using CUDA.CUFFT, Flux

"""
    function gpu_power_kernel!(Π_X_d, Π, d, M)

    Kernel function to compute the polynomial multiplication on the GPU.

    Args:
        Π_X_d::CuArray{Float64}: The output array for the polynomial coefficients.
        Π::CuArray{Float64}: The input polynomial coefficients.
        d::Int64: The exponent to which the polynomial is raised.
        M::Int64: The size of the polynomial.
"""
function gpu_power_kernel!(Π_X_d, Π, d, M)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && b <= size(Π, 2)
        new_idx = mod((i - 1) * d, M) + 1
        @inbounds Π_X_d[new_idx, b] += Π[i, b]
    end

    return nothing
end

function power_gpu(Π::CuArray{Float64}, d::Int64)
    M, batch_size = size(Π)
    Π_X_d = CUDA.zeros(Float64, M, batch_size)

    threads = (32, 8)
    blocks = (cld(M, threads[1]), cld(batch_size, threads[2]))

    @cuda threads=threads blocks=blocks gpu_power_kernel!(Π_X_d, Π, d, M)

    return complex.(Π_X_d)
end

# Custom kernel: rotate each column of Π by the same amount `ind`
function kernel_rotate_poly_add!(P, Π, ind::Int, M)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && b <= size(Π, 2)
        src_i = mod1(i - ind, M)
        P[i, b] += Π[src_i, b]
    end
    return
end

function batch_rotate_poly_gpu!(Π_acc::CuMatrix{Float64},Π_batch::CuMatrix{Float64}, ind::Int)
    M, B = size(Π_batch)

    threads = (16, 16)
    blocks = (cld(M, threads[1]), cld(B, threads[2]))

    @cuda threads=threads blocks=blocks kernel_rotate_poly_add!(Π_acc, Π_batch, ind, M)

    return
end

function fused_rotate_pack!(
    a_new::CuDeviceMatrix{Float64}, b_new::CuDeviceMatrix{Float64},
    a_0::CuDeviceMatrix{Float64}, b_0::CuDeviceMatrix{Float64},
    M::Int, t::Int, β::Int, output_size::Int, new_output_size::Int, N::Int
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    pck_size = t^β * (N ÷ M) 
    tβ1 = t^(β - 1)

    while idx <= new_output_size
        pck_idx = (idx - 1) ÷ tβ1 + 1
        i = mod(idx - 1, tβ1) + 1
        new_pck_idx = (pck_idx - 1) * tβ1
        dest_col = new_pck_idx + i

        @inbounds for m in 0:M-1
            acc_a = 0.0
            acc_b = 0.0

            for j in 0:(t - 2)
                linear_ind = i + j * tβ1 + (pck_idx - 1) * pck_size
                if linear_ind <= output_size
                    offset = mod1(j * div(M, t), M)
                    src_idx = mod(m - offset, M)

                    acc_a += a_0[src_idx + 1, linear_ind]
                    acc_b += b_0[src_idx + 1, linear_ind]
                end
            end

            a_new[m + 1, dest_col] = acc_a
            b_new[m + 1, dest_col] = acc_b
        end

        idx += stride
    end

    return
end

function fused_add_rotate!(
    a_new::CuDeviceMatrix{Float64}, b_new::CuDeviceMatrix{Float64},
    a_in::CuDeviceMatrix{Float64}, b_in::CuDeviceMatrix{Float64},
    M::Int, t::Int, β::Int, j::Int,
    output_size::Int, nb_pck::Int
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    tβ_j = t^(β - j)
    pck_size = t^(β - j + 1)
    new_output_size = tβ_j * nb_pck

    while idx <= new_output_size
        pck_idx = (idx - 1) ÷ tβ_j + 1
        i = mod(idx - 1, tβ_j) + 1
        new_pck_idx = (pck_idx - 1) * tβ_j
        dest_col = new_pck_idx + i

        # Temporary registers to accumulate result per row
        @inbounds for m in 0:M-1
            acc_a = 0.0f0
            acc_b = 0.0f0

            for i_β in 0:(t - 1)
                linear_ind = (pck_idx - 1) * pck_size + i_β * tβ_j + i
                if linear_ind <= output_size
                    rot_offset = mod1(i_β * div(M, t^j), M)
                    src_idx = mod(m - rot_offset, M)

                    acc_a += a_in[src_idx + 1, linear_ind]
                    acc_b += b_in[src_idx + 1, linear_ind]
                end
            end

            a_new[m + 1, dest_col] = acc_a
            b_new[m + 1, dest_col] = acc_b
        end

        idx += stride
    end
    return
end