"""
    resample(p::Particles, essthresh, rs)

Resamples the particle object `p` if the ess is under `essthresh`. The
resampling algorithm `rs` is for example a multinomial resampling.
"""
function resample(p::Particles, essthresh::Float=Inf,
                    rs::Function=multinomialresampling, M::Int=0
                    )::Tuple{Particles,Float}
    ess = 1.0/sum(p.w.^2)
    N   = length(p)
    M   = M>0 ? M : N
    (M != N || ess < essthresh * N) ? (rs(p, M), ess) : (p, ess)
end

"""
    multinomialresampling(p::Particles)

Multinomial resampling of a particles object `p`.
"""
function multinomialresampling(p::Particles, M::Int=0)::Particles
    N    = length(p)
    M    = (M>0)? M : N
    ni   = rand(Multinomial(M, p.w))
    mask = [j for i in 1:N for j in ones(Int,ni[i])*i]
    Particles(p.x[mask], ones(M)/M)
end
