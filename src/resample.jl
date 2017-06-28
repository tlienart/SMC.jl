"""
    resample(p::Particles, essthresh, rs)

Resamples the particle object `p` if the ess is under `essthresh`. The
resampling algorithm `rs` is for example a multinomial resampling.
"""
function resample(p::Particles, essthresh::Float=Inf,
                    rs::Function=multinomialresampling
                    )::Tuple{Particles,Float}
    ess = 1.0/sum(p.w.^2)
    (ess < essthresh * length(p)) ? (rs(p),ess) : (p,ess)
end

"""
    multinomialresampling(p::Particles)

Multinomial resampling of a particles object `p`.
"""
function multinomialresampling(p::Particles)::Particles
    N    = length(p)
    ni   = rand(Multinomial(N, p.w))
    mask = [j for i in 1:N for j in ones(Int,ni[i])*i]
    Particles(p.x[mask], ones(N)/N)
end
