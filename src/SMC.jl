module SMC

using Compat
using Distributions:Multinomial

import Base.length, Base.mean

const Int   = Int64
const Float = Float64

include("hmm.jl")
include("kalman.jl")

include("particleset.jl")
include("particlefilter.jl")
include("particlesmoother.jl")

end # module
