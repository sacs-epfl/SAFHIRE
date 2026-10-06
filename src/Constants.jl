##################
### Parameters ###
##################
# Configurable parameters: set _param_* variables before including this file.
# If not set, defaults match the 16-bit precision, γ∈[1,2] configuration.
#
const S=[-1;1]            # The subset of Z in which the components of the key are picked
const σ_LWE=2^(-26)       # Standard deviation of the Gaussian noise in LWE-encryption
const σ_RLWE= @isdefined(_param_σ_RLWE) ? _param_σ_RLWE : 2^(-51)
const p= @isdefined(_param_p) ? _param_p : 2^16
#
#
##############################
### Cyclotomic index M=t^α ###
##############################
#
const α= @isdefined(_param_α) ? _param_α : 5
const t= @isdefined(_param_t) ? _param_t : 5
const g= @isdefined(_param_g) ? _param_g : 2
const M=t^α               # Cyclotomic index
const N=t^(α-1)*(t-1)     # The Euler function of M
const n=N                 # LWE-key dimension (set to Euler totient)
const M_over_t=t^(α-1)    # Used for change of bases
const M_inv = mod(gcdx(M, p)[2], p)             # The invert of M mod p
const D = @isdefined(_param_D) ? _param_D : 310
const l = @isdefined(_param_l) ? _param_l : 4
const β=α-1               # Packing level (derived)

const one_t=ones(t)             # Vector of size t with unit components
const F=1.0/M                   # F=1/M used in the change of bases
const Ω0_star_inv=zeros(M)      # Innverse of Ω^*_0(X) = (1-X^N)/M expressed in the basis Ω_i
Ω0_star_inv[1]=M_over_t
for k in 1:t-2
    Ω0_star_inv[1+k*M_over_t]=M_over_t*(k+1-t)
end
####################
### Plan for FFT ###
####################
#
# Sets the number of threads to FFTW
FFTW.set_num_threads(1)   
const a_M=Vector{ComplexF64}(undef,M)                                 # Model vector of size M
const op_fft_Mi=plan_fft!(a_M; flags=FFTW.ESTIMATE, timelimit=Inf)    # Prepares M-dimensional Fourier in place
const op_ifft_Mi=plan_ifft!(a_M; flags=FFTW.ESTIMATE, timelimit=Inf)  # Prepares M-dimensional inverse Fourier in place
const op_fft_M=plan_fft(a_M; flags=FFTW.MEASURE, timelimit=Inf)      # Prepares M-dimensional Fourier
const op_ifft_M=plan_ifft(a_M; flags=FFTW.MEASURE, timelimit=Inf)    # Prepares M-dimensional inverse Fourier
const FFT_Ω0_star_inv=op_fft_M*(Ω0_star_inv)                          # Defines the Fourier of (Ω^*0)^{-1} 
#
###################
### Secret keys ###
###################
#
const Key_LWE=LWE_Key(undef,n)
Key_LWE=rand(S,n)
const Key_RLWE=RLWE_Key(undef,M)
Key_RLWE=[rand(S,N);Int64.(zeros(M-N))]
const FFT_Key_RLWE=FFT_Key(undef,M)
FFT_Key_RLWE=op_fft_M*(Key_RLWE)
#