module MyPackage

    # External dependencies used widely
    using FFTW

    include("Constants.jl")
    include("Types.jl")
    include("utils/BaseFunctions.jl")
    include("utils/Operations.jl")
    include("utils/Encryption.jl")
    include("utils/KeySwitches.jl")
    include("PartialTraces.jl")
    include("Packing.jl")

    

end