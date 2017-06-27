export
    ParticleSet,
    multinomialresampling

const ParticleType = Union{Float, Vector{Float}}

mutable struct ParticleSet{T<:ParticleType}
    p::Vector{T}
    w::Vector{Float}
end

length(ps::ParticleSet) = length(ps.w)
mean(ps::ParticleSet)   = sum(ps.p[i] * ps.w[i] for i = 1:length(ps))

function multinomialresampling(ps::ParticleSet)::ParticleSet
    N    = length(ps)
    ni   = rand(Multinomial(length(ps), ps.w))
    mask = [j for i in 1:N for j in ones(Int,ni[i])*i]
    ParticleSet(ps.p[mask], ones(N)/N)
end
