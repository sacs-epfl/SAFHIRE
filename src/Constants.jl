##################
### Parameters ###
##################
#
const n=1331              # Number of components in the secret LWE-key
const S=[-1;1]            # The subset of Z in which the components of the key are picked
const σ_LWE=2^(-26)       # Standard deviation of the Gaussian noise in LWE-encryption
const σ_RLWE=2^(-51)      # Standard deviation of the Gaussian noise in RLWE-encryption
const p= 2^16 #65624 #781       # Number of elements in the discrete torus T_p
#
#
##############################
### Cyclotomic index M=t^α ###         
##############################
#
const α=5                 # A strictly positive integer 
const t=5                # A prime number 
const g=2                  # Generator of (Z/MZ,x)
const M=t^α               # Cyclotomic index
const N=t^(α-1)*(t-1)     # The Euler function of M
const M_over_t=t^(α-1)    # Used for change of bases
const M_inv = mod(gcdx(M, p)[2], p)             # The invert of M mod p
const D = 310
const l = 4
const β=4                 # Packing level

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
# Sets the number of threads to 1 for FFTW, avoid segmentation faults when modifying M.
# Also if the plan_fft is called with one thread it will be faster approx 3x faster than if
# set_num_threads was set to 8. Therefore we advise to multithread loops where each step uses FFT rather 
# than multithreading the FFT itself.
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