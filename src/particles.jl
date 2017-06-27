export
    Particles,
    ParticleSet,
    multinomialresampling

const ParticleType = Union{Float, Vector{Float}}

mutable struct Particles{T <: ParticleType}
    x::Vector{T}     # N x (1) or N x (dimx)
    w::Vector{Float} # N
end
Particles(N::Int, dx::Int) = Particles(
    dx==1 ? zeros(N) : Vector{Vector{Float}}(N), ones(N)/N)

mutable struct ParticleSet{T <: ParticleType}
    p::Vector{Particles{T}} # T
end
ParticleSet(N::Int, dx::Int, K::Int) =
    ParticleSet([Particles(N,dx) for i in 1:K])

length(p::Particles)    = length(p.w)
length(ps::ParticleSet) = length(ps.p)

mean(p::Particles) = sum(p.x[i] * p.w[i] for i = 1:length(p))

function multinomialresampling(p::Particles, essthresh::Float=Inf)::Particles
    N    = length(p)
    ni   = rand(Multinomial(N, p.w))
    mask = [j for i in 1:N for j in ones(Int,ni[i])*i]
    Particles(p.x[mask], ones(N)/N)
end

function resample(p::Particles, essthresh::Float=0.0,
                    rs::Function=multinomialresampling
                    )::Tuple{Particles,Float}
    ess = 1.0/sum(p.w.^2)
    (ess < essthresh * length(p)) ? (rs(p),ess) : (p,ess)
end
