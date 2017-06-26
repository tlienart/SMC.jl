export
    LinearGaussian,
    generate

abstract type HMM end

struct LinearGaussian <: HMM
    #=
        x <- Ax + Q*randn
        y <- Bx + R*randn
    =#
    A::Matrix{Float}
    B::Matrix{Float}
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    # implicit
    dimx::Int
    dimy::Int
    function LinearGaussian(A,B,Q,R)
        dimx=size(A,1)
        dimy=size(B,1)
        @assert dimx==size(A,2)==size(Q,1)==size(Q,2) "dimensions don't match"
        @assert dimy==size(R,1)==size(R,2) "dimensions don't match"
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new(A,B,chol(Q),chol(R), dimx, dimy)
    end
end

"""
    generate(hmm, x0, T)

Generate observations following a given dynamic for `T` time steps.
"""
function generate(hmm::LinearGaussian, x0::Vector{Float}, T::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==hmm.dimx "dimensions don't match"

    states, observations = zeros(hmm.dimx, T), zeros(hmm.dimy, T)

    states[:,1] = x0

    noisex = hmm.cholQ'*randn(hmm.dimx,T)
    noisey = hmm.cholR'*randn(hmm.dimy,T)

    for t = 1:(T-1)
        observations[:,t] = hmm.B*states[:,t] + noisey[:,t]
        states[:,t+1]     = hmm.A*states[:,t] + noisex[:,t+1]
    end
    observations[:,T] = hmm.B*states[:,T] + noisey[:,T]

    return (states, observations)
end
