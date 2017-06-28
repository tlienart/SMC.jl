module SMC

using Compat
# using ExpFamily
using Distributions


import Base.length, Base.mean

const Int   = Int64
const Float = Float64

include("hmm.jl")
include("kalman.jl")

include("particles.jl")
include("resample.jl")
include("particlefilter.jl")
include("particlesmoother.jl")

end # module
