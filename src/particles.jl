export
    Particles,
    ParticleSet,
    multinomialresampling

const ParticleType = Union{Float, Vector{Float}}

mutable struct Particles{T <: ParticleType}
    x::Vector{T}     # N x (1) or N x (dimx)
    w::Vector{Float} # N
end

mutable struct ParticleSet{T <: ParticleType}
    p::Vector{Particles{T}} # T
end


"""
    Particles(N,dx)

Create a Particles object with `N` particles each of dimension `dx`.
"""
Particles(N::Int, dx::Int=1) =
    Particles( dx==1 ? zeros(N) : Vector{Vector{Float}}(N), ones(N)/N )

"""
    ParticleSet(N,dx,K)

Create a set of `K` Particles with `N` particles of dimension `dx`. This is for
a HMM with `K` steps.
"""
ParticleSet(N::Int, dx::Int, K::Int) =
    ParticleSet( [Particles(N,dx) for i in 1:K] )

"""
    length(p::Particles)

Number of particles.
"""
length(p::Particles)    = length(p.w)

"""
    length(ps::ParticleSet)

Number of slices (steps in the HMM).
"""
length(ps::ParticleSet) = length(ps.p)

"""
    mean(p::Particles)

Compute the mean corresponding to the particles `p`.
"""
mean(p::Particles) = sum(p.x[i] * p.w[i] for i = 1:length(p))

"""
    mean(ps::Particles)

Compute the mean corresponding to a particle set
"""
mean(ps::ParticleSet) = [mean(p) for p in ps.p]
