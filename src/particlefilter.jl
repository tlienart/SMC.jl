export particlefilter

function particlefilter(hmm::HMM, observations::Matrix{Float}, N::Int,
                        proposal::Proposal;
                        resampling::Function=multinomialresampling,
                        essthresh::Float=0.5
                        )::Tuple{ParticleSet,Vector{Float}}

    K   = size(observations, 2)
    # particle set filter (storage)
    psf = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)

    (p1,e1) = resample(
                Particles( [proposal.noise() for i in 1:N], ones(N)/N),
                essthresh )
    # store
    psf.p[1] = p1
    ess[1]   = e1

    for k=2:K
        pkm1 = psf.p[k-1]
        obsk = observations[:,k]

        logak = zeros(N)
        xk    = similar(pkm1.x)
        # sample (BOOTSTRAP)
        for i in 1:N
            xk[i]    = proposal.mean(k, pkm1.x[i], obsk) + proposal.noise()
            logak[i] = hmm.transloglik(k, pkm1.x[i], xk[i]) +
                        hmm.obsloglik(k, xk[i], obsk) -
                        proposal.loglik(k, pkm1.x[i], obsk, xk[i])
        end

        Wk  = log.(pkm1.w) + logak
        Wk -= minimum(Wk) # try to avoid underflows
        wk  = exp.(Wk)
        wk /= sum(wk)

        (pk, ek) = resample(Particles(xk,wk), essthresh, resampling)

        psf.p[k] = pk
        ess[k]   = ek
    end
    (psf, ess)
end
