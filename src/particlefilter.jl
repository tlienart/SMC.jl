export particlefilter

function particlefilter(hmm::HMM, observations::Matrix{Float}, N::Int,
#                        samplemu1::Function,
                        essthresh::Float=0.5
                        )::Tuple{ParticleSet,Vector{Float}}

    K   = size(observations, 2)
    ps  = ParticleSet(N, hmm.dimx, K)
    ess = zeros(N)

#    (p1,e1) = resample(Particles(samplesmu1(N), ones(N)/N), essthresh)
    (p1,e1) = resample(Particles(
                        [hmm.transnoise() for i in 1:N], ones(N)/N), essthresh)
    ps.p[1] = p1
    ess[1]  = e1

    for k=2:K
        pkm1 = ps.p[k-1]
        obsk = observations[:,k]

        logak = zeros(N)
        xk    = similar(pkm1.x)
        # sample (BOOTSTRAP)
        for i in 1:N
            xk[i]    = hmm.transmean(k, pkm1.x[i]) + hmm.transnoise()
            logak[i] = hmm.obsloglik(k, xk[i], obsk)
        end

        Wk  = log.(pkm1.w) + logak
        Wk -= minimum(Wk) # try to avoid underflows
        wk  = exp.(Wk)
        wk /= sum(wk)

        (pk, ek) = resample(Particles(xk,wk), essthresh)

        ps.p[k] = pk
        ess[k]  = ek
    end
    (ps, ess)
end
