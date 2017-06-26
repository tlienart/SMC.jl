module SMC

using Compat

const Int   = Int64
const Float = Float64

include("hmm.jl")
include("kalman.jl")

end # module
