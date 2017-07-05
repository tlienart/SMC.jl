export
    particlesmoother_ffbs,
    particlesmoother_bbis
"""
    particlesmoother_ffbs

Particle smoother based on the Forward Filtering Backward Smoothing algorithm.
The complexity of this algorithm is O(KN^2)
"""
function particlesmoother_ffbs(hmm::HMM, psf::ParticleSet)
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    psw = deepcopy(psf)
    #
    pk = psw.p[K]

    for k=(K-1):-1:1
        pkp1 = pk
        pk   = psw.p[k]
        # denominator precomputation: O(N^2) comp, O(N) storage
        ds = [ sum( pk.w[l] *
                      exp(hmm.transloglik(k,pk.x[l],pkp1.x[j]))
                        for l in 1:N ) for j in 1:N ]
        # FFBS formula, in place computations, O(N) for each i so O(N^2) all
        for i in 1:N
            pk.w[i] *= sum( pkp1.w[j] *
                              exp(hmm.transloglik(k,pk.x[i],pkp1.x[j])) / ds[j]
                                for j in 1:N )
        end
        # normalisation
        pk.w /= sum(pk.w)
        # store the updated weights
        psw.p[k].w = copy(pk.w)
    end
    psw
end

"""
  particlesmoother_bbis

Bootstrap backward information smoother.
"""
function particlesmoother_bbis(hmm::HMM, observations::Matrix{Float}, M::Int,
                               psf::ParticleSet, bootstrap::Proposal;
                               resampling::Function=multinomialresampling,
                               essthresh::Float=0.5
                               )::Tuple{ParticleSet,Vector{Float}}
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    pss = ParticleSet(M, hmm.dimx, K)
    ess = zeros(K)
    # Particles at last step need to be resampled
    (pK,eK)  = resample(psf.p[K], essthresh, resampling, M)
    pss.p[K] = pK
    ess[K]   = eK
    #
    for k=(K-1):-1:2
        obsk   = observations[:,k]
        psskp1 = pss.p[k+1] # smoothing particles from previous step (k+1)
        psfk   = psf.p[k]   # filtering particles for PD_{k+1}(x_k+1)
        psfkm1 = psf.p[k-1] # filtering particles for PD_{k}(x_k)
        # preparing indices (for each j sample from 1 of the mixture component)
        randmult = rand(Multinomial(N, psfkm1.w))
        indices  = [i for k in 1:N for i in ones(Int,randmult[k])*k] # unroll
        # precompute denominator of the update factor PD_{k+1}(x_k+1)
        denj = [ sum( psfk.w[l] *
                        exp(hmm.transloglik(l, psfk.x[l], psskp1.x[j]))
                          for l in 1:N ) for j in 1:M ]
        xk    = similar(psskp1.x)
        logak = zeros(M)
        # sample and update weight for each smoothing particle
        for j = 1:M
            # sampling from corresponding element (see multinomial step)
            xk[j] = bootstrap.mean(k, psfkm1.x[indices[j]]) + bootstrap.noise()
            # weight update factor
            logak[j] = hmm.transloglik(k, xk[j], psskp1.x[j]) +
                        hmm.obsloglik(k, xk[j], obsk) -
                          log(denj[j])
        end
        # normalise weights
        Wk  = log.(psskp1.w) + logak
        Wk -= minimum(Wk)
        wk  = exp.(Wk)
        wk /= sum(wk)

        (pk, ek) = resample(Particles(xk, wk), essthresh, resampling)

        pss.p[k] = pk
        ess[k]   = ek
    end

    # ----------------------------------------
    # LAST STEP (sampling from gamma1 = prior)

    obsk   = observations[:,1]
    psskp1 = pss.p[2] # smoothing particles from previous step (k+1)
    psfk   = psf.p[1]   # filtering particles for PD_{k+1}(x_k+1)
    denj   = [ sum( psfk.w[l] *
                      exp(hmm.transloglik(l, psfk.x[l], psskp1.x[j]))
                        for l in 1:N ) for j in 1:M ]
    xk     = similar(psskp1.x)
    logak  = zeros(M)

    for j = 1:M
        xk[j]    = bootstrap.noise() # sampling from prior
        logak[j] = hmm.transloglik(1, xk[j], psskp1.x[j]) +
                    hmm.obsloglik(1, xk[j], obsk) -
                      log(denj[j])
    end
    # normalise weights
    Wk  = log.(psskp1.w) + logak
    Wk -= minimum(Wk)
    wk  = exp.(Wk)
    wk /= sum(wk)

    (pss.p[1], ess[1]) = resample(Particles(xk,wk), essthresh, resampling)

    (pss, ess)
end
