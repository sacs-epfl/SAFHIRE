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