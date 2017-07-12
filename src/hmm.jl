export
    HMM,
    LinearGaussian,
    NonLinearGaussian,
    GaussianHMM,
    generate

abstract type AbstractHMM end
abstract type GaussianHMM end

struct HMM <: AbstractHMM
    transmean::Function
    transloglik::Function # log transition function:  f(xk|xkm1)
    obsmean::Function
    obsloglik::Function   # log observation function: g(y|xk)
    dimx::Int
    dimy::Int
end

struct LinearGaussian <: GaussianHMM
    #=
        x <- Ax + chol(Q)' * randn
        y <- Bx + chol(R)' * randn
    =#
    A::Matrix{Float}
    B::Matrix{Float}
    Q::Union{Float,Matrix{Float}}
    R::Union{Float,Matrix{Float}}
    # implicit
    dimx::Int
    dimy::Int
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    transmean::Function
    obsmean::Function
    function LinearGaussian(A,B,Q,R)
        dimx=size(A,1)
        dimy=size(B,1)
        @assert dimx==size(A,2)==size(Q,1)==size(Q,2) "dimensions don't match"
        @assert dimy==size(R,1)==size(R,2) "dimensions don't match"
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new(A,B,Q,R,dimx,dimy,chol(Q),chol(R),
            (k,xkm1) -> A*xkm1,
            (k,xk)   -> B*xk )
    end
end

struct NonLinearGaussian <: GaussianHMM
    #=
        x <- f(x) + chol(Q)' * randn
        y <- g(x) + chol(R)' * randn
    =#
    transmean::Function
    obsmean::Function
    Q::Union{Float,Matrix{Float}}
    R::Union{Float,Matrix{Float}}
    # implicit
    dimx::Int
    dimy::Int
    cholQ::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    cholR::Union{Float,UpperTriangular{Float, Matrix{Float}}}
    function NonLinearGaussian(f, g, Q, R)
        dimx = size(Q, 1)
        dimy = size(R, 1)
        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"
        new((k,xkm1)-> f(k,xkm1),
            (k,xk)  -> g(k,xk),
            Q,R,dimx,dimy,chol(Q),chol(R)  )
    end
end

function HMM(g::GaussianHMM)
    transloglik = (k,xkm1,xk) -> -norm(g.cholQ'\(xk - g.transmean(k,xkm1)))^2/2
    obsloglik   = (k,yk,  xk) -> -norm(g.cholR'\(yk - g.obsmean(k,xk)))^2/2
    HMM(g.transmean, transloglik, g.obsmean, obsloglik, g.dimx, g.dimy)
end

### Generation of observations

"""
    generate(lg, x0, K)

Generate observations following a given (non)linear Gaussian dynamic for `K`
time steps.
"""
function generate(g::GaussianHMM, x0::Vector{Float}, K::Int
                    )::Tuple{Matrix{Float},Matrix{Float}}
    @assert length(x0)==g.dimx "dimensions don't match"
    # allocate states/observations
    states, observations = zeros(g.dimx, K), zeros(g.dimy, K)
    # assign first state
    states[:,1] = x0
    # pre-generate noise
    noisex = g.cholQ' * randn(g.dimx,K)
    noisey = g.cholR' * randn(g.dimy,K)
    # use noise in iterative linear system
    for k = 1:(K-1)
        observations[:,k] = g.obsmean(k, states[:,k]) + noisey[:,k]
        states[:,k+1]     = g.transmean(k+1, states[:,k]) + noisex[:,k+1]
    end
    # last observation
    observations[:,K] = g.obsmean(K, states[:,K]) + noisey[:,K]
    # package and return
    return (states, observations)
end
