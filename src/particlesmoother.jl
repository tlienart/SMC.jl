export
    particlesmoother_ffbs,
    particlesmoother_lffbs,
    particlesmoother_bbis,
    particlesmoother_lbbis,
    particlesmoother_llbbis,
    particlesmoother_fearnhead_lg

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
        # denominator precomputation
        # -- complexity O(N^2) (linear storage)
        ds = [ sum( pk.w[l] *
                      exp(hmm.transloglik(k,pk.x[l],pkp1.x[j]))
                        for l in 1:N ) for j in 1:N ]
        # FFBS formula, in place computations
        # -- complexity O(N^2)
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
    particlesmoother_lffbs

Particle smoother based on the Forward Filtering Backward Smoothing algorithm.
The complexity of this algorithm is O(KNM)
"""
function particlesmoother_lffbs(hmm::HMM, psf::ParticleSet, M::Int)
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    psw = deepcopy(psf)
    #
    pk = psw.p[K]

    for k=(K-1):-1:1
        pkp1 = pk
        pk   = psw.p[k]

        r = rand(Multinomial(M, pkp1.w))
        r = [i for k in 1:N for i in ones(Int,r[k])*k]

        # denominator precomputation
        # -- complexity O(NM) (linear storage)
        ds = [ sum( pk.w[l] *
                      exp(hmm.transloglik(k,pk.x[l],pkp1.x[r[j]]))
                        for l in 1:N ) for j in 1:M ]
        # FFBS formula, in place computations
        # -- complexity O(NM)
        for i in 1:N
            pk.w[i] *= sum( pkp1.w[r[j]] *
                              exp(hmm.transloglik(k,pk.x[i],pkp1.x[r[j]])) / ds[j]
                                for j in 1:M )
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

Bootstrap backward information smoother. (CONSISTENT)
"""
function particlesmoother_bbis(hmm::HMM, observations::Matrix{Float},
                               psf::ParticleSet, bootstrap::Proposal;
                               resampling::Function=multinomialresampling,
                               essthresh::Float=0.5
                               )::Tuple{ParticleSet,Vector{Float}}
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    pss = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    # Particles at last step need to be resampled
    (pK,eK)  = resample(psf.p[K], essthresh, resampling, N)
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
        # -- complexity O(N^2) (linear storage)
        denj = [ sum( psfk.w[l] *
                        exp(hmm.transloglik(l, psfk.x[l], psskp1.x[j]))
                          for l in 1:N ) for j in 1:N ]
        xk    = similar(psskp1.x)
        logak = zeros(N)
        # sample and update weight for each smoothing particle
        # -- complexity O(N)
        for j = 1:N
            # sampling from corresponding element (see multinomial step)
            xk[j] = bootstrap.mean(k, psfkm1.x[indices[j]]) + bootstrap.noise()
            # weight update factor
            logak[j] = hmm.transloglik(k, xk[j], psskp1.x[j]) +
                        hmm.obsloglik(k, obsk, xk[j]) -
                          log(denj[j])
        end
        # normalise weights
        # -- complexity O(N)
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
                        for l in 1:N ) for j in 1:N ]
    xk     = similar(psskp1.x)
    logak  = zeros(N)

    for j = 1:N
        xk[j]    = bootstrap.mu0 + bootstrap.noise() # sampling from prior
        logak[j] = hmm.transloglik(1, xk[j], psskp1.x[j]) +
                    hmm.obsloglik(1, obsk, xk[j]) -
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

"""
  particlesmoother_lbbis

Bootstrap backward information smoother with linear complexity (APPROX BBIS).
"""
function particlesmoother_lbbis(hmm::HMM, observations::Matrix{Float},
                                psf::ParticleSet, bootstrap::Proposal;
                                resampling::Function=multinomialresampling,
                                essthresh::Float=0.5
                                )::Tuple{ParticleSet,Vector{Float}}
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    pss = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    # Particles at last step need to be resampled
    (pK,eK)  = resample(psf.p[K], essthresh, resampling, N)
    pss.p[K] = pK
    ess[K]   = eK
    #
    for k=(K-1):-1:2
        obsk   = observations[:,k]
        psskp1 = pss.p[k+1] # smoothing particles from previous step (k+1)
        psfk   = psf.p[k]   # filtering particles for PD_{k+1}(x_k+1)
        psfkm1 = psf.p[k-1] # filtering particles for PD_{k}(x_k)

        # preparing indices for the forward part
        randmult_fwd = rand(Multinomial(N, psfkm1.w))
        randmult_bwd = rand(Multinomial(N, psskp1.w))
        # unroll indices into masks
        indices_fwd = [i for k in 1:N for i in ones(Int,randmult_fwd[k])*k]
        indices_bwd = [i for k in 1:N for i in ones(Int,randmult_bwd[k])*k]

        xk    = similar(psskp1.x)
        logak = zeros(N)
        # sample and update weight for each smoothing particle
        for j = 1:N
            # sampling from corresponding element (see multinomial step)
            xk[j] = bootstrap.mean(k, psfkm1.x[indices_fwd[j]]) +
                      bootstrap.noise()
            # weight update factor p(x_t+1|x_t)p(y_t|x_t)
            logak[j] = hmm.transloglik(k, xk[j], psskp1.x[indices_bwd[j]]) +
                         hmm.obsloglik(k, obsk, xk[j])
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

    randmult_bwd = rand(Multinomial(N, psskp1.w ))
    indices_bwd  = [i for k in 1:N for i in ones(Int,randmult_bwd[k])*k]

    xk     = similar(psskp1.x)
    logak  = zeros(N)

    for j = 1:N
        xk[j]    = bootstrap.mu0 + bootstrap.noise() # sampling from prior
        logak[j] = hmm.transloglik(1, xk[j], psskp1.x[indices_bwd[j]]) +
                    hmm.obsloglik(1, obsk, xk[j])
    end
    # normalise weights
    Wk  = log.(psskp1.w) + logak
    Wk -= minimum(Wk)
    wk  = exp.(Wk)
    wk /= sum(wk)

    (pss.p[1], ess[1]) = resample(Particles(xk,wk), essthresh, resampling)

    (pss, ess)
end

"""
  particlesmoother_llbbis

Bootstrap backward information smoother with log-log resampling (CONSISTENT)
"""
function particlesmoother_llbbis(hmm::HMM, observations::Matrix{Float},
                                 psf::ParticleSet, M::Int, bootstrap::Proposal;
                                 resampling::Function=multinomialresampling,
                                 essthresh::Float=0.5
                                 )::Tuple{ParticleSet,Vector{Float}}
    K = length(psf)
    N = length(psf.p[1])
    # particle set smoother (storage)
    pss = ParticleSet(N, hmm.dimx, K)
    ess = zeros(K)
    # Particles at last step need to be resampled
    (pK,eK)  = resample(psf.p[K], essthresh, resampling, N)
    pss.p[K] = pK
    ess[K]   = eK
    # force sub-resampling of forward components
    (psfkm1, e) = resample(psf.p[K-1], Inf, resampling, M)
    #
    for k=(K-1):-1:2
        obsk   = observations[:,k]
        # smoothing particles from previous step (k+1) + force sub-resampling
        psskp1 = pss.p[k+1]
        # filtering particles for PD_{k+1}(x_k+1) + recycled from previous step
        psfk   = psfkm1
        # filtering particles for PD_{k}(x_k)
        (psfkm1, e) = resample(psf.p[k-1], Inf, resampling, M)
        # preparing indices (for each j sample from 1 of the mixture component)
        # since we've done the resampling all mixture components have equal w
        randmult = rand(Multinomial(N, M))
        indices  = [i for k in 1:M for i in ones(Int,randmult[k])*k] # unroll
        # precompute denominator of the update factor PD_{k+1}(x_k+1)
        # -- complexity O(N*M) (N storage)
        denj = [ sum( psfk.w[l] *
                        exp(hmm.transloglik(l, psfk.x[l], psskp1.x[j]))
                          for l in 1:M ) for j in 1:N ]
        xk    = similar(psskp1.x)
        logak = zeros(N)
        # sample and update weight for each smoothing particle
        # -- complexity O(N)
        for j = 1:N
            # sampling from corresponding element (see multinomial step)
            xk[j] = bootstrap.mean(k, psfkm1.x[indices[j]]) + bootstrap.noise()
            # weight update factor
            logak[j] = hmm.transloglik(k, xk[j], psskp1.x[j]) +
                        hmm.obsloglik(k, obsk, xk[j])
        end
        # normalise weights
        # -- complexity O(N)
        Wk  = log.(psskp1.w) + logak - log.(denj)
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
    psfk   = psfkm1   # filtering particles for PD_{k+1}(x_k+1)
    denj   = [ sum( psfk.w[l] *
                      exp(hmm.transloglik(l, psfk.x[l], psskp1.x[j]))
                        for l in 1:M ) for j in 1:N ]
    xk     = similar(psskp1.x)
    logak  = zeros(N)

    for j = 1:N
        xk[j]    = bootstrap.noise() # sampling from prior
        logak[j] = hmm.transloglik(1, xk[j], psskp1.x[j]) +
                    hmm.obsloglik(1, obsk, xk[j]) -
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


"""
    particlesmoother_fearnhead_lg

Linear complexity smoother a la Fearnhead with normalising densities obtained
via the simple dynamic. This is just for reproduction purposes and should
not really be used.
"""
function particlesmoother_fearnhead_lg(lg::LinearGaussian,
                                       observations::Matrix{Float},
                                       psf::ParticleSet;
                                       resampling=multinomialresampling,
                                       essthresh::Float=0.5)::ParticleSet
    K = length(psf)
    N = length(psf.p[1])

    Qi   = inv(lg.Q)
    QiA  = Qi * lg.A
    AQiA = lg.A' * QiA

    Ri   = inv(lg.R)
    RiB  = Ri * lg.B
    BRiB = lg.B' * RiB

    ############################################
    # Computation of the means and covariances
    # of the normalising densities γ_k
    ############################################
    # initialise
    gamma_mu = zeros(lg.dimx, K)
    gamma_S  = zeros(lg.dimx, lg.dimx, K)

    gamma_mu[:, 1] = zeros(lg.dimx)    # this could be passed to function
    gamma_S[:,:,1] = 0.2*eye(lg.dimx)  # this could be passed to function

    for k = 2:K
        # Cf thesis appendix for full development, just compl of quadratics
        temp_S         = inv(inv(gamma_S[:,:,k-1]) + AQiA)
        gamma_S[:,:,k] = inv(Qi - QiA * temp_S * QiA')
        temp_mu        = gamma_S[:,:,k-1] \ gamma_mu[:,k-1]
        gamma_mu[:,k]  = gamma_S[:,:,k] * QiA * temp_S * temp_mu
    end
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    ############################################
    # Targeting the Backward Information Filter
    # with the normalising densities computed
    # above. Since everything is tractable,
    # proposal is the optimal proposal.
    ############################################

    bif = ParticleSet(N, lg.dimx, K)

    # initial: sample from γ_K p(y_K | x_K), completion of squares again

    yK      = observations[:,K]
    bif_SK  = inv( inv(gamma_S[:,:,K]) + BRiB )
    bif_muK = bif_SK * ( (gamma_S[:,:,K] \ gamma_mu[:, K]) + RiB' * yK )

    noise    = chol(Symmetric(bif_SK))' * randn(lg.dimx, N)
    bif.p[K] = Particles( [ bif_muK + noise[:,j] for j in 1:N] ,
                          ones(N)/N)

    mvnpdf(x,mu,S) = exp(-0.5dot(S\(x-mu),x-mu))
    #
    for k = K-1:-1:1
        # observation
        yk = observations[:,k]

        # construct the gaussian γ_t * p(x_k+1 | x_k) p(y_k | x_k)
        # > covariance
        optpropS   = Symmetric(inv( inv(gamma_S[:,:,k]) + BRiB + AQiA ))
        # > mean (fixed part not depending on x_k+1)
        optpropmu_ = optpropS * ( gamma_S[:,:,k] \ gamma_mu[:,k] + RiB' * yk )
        # > mean (variable part, depending on x_k+1)
        optpropmu_j = [optpropS * QiA' * bif.p[k+1].x[j] for j in 1:N]

        # sample from optimal prop
        noise   = chol(optpropS)' * randn(lg.dimx, N)
        samples = [optpropmu_ + optpropmu_j[j] + noise[:,j] for j in 1:N]

        # compute γ_k+1 for each x_k+1
        gammakp1 = [ mvnpdf( bif.p[k+1].x[j], gamma_mu[:,k+1],
                       gamma_S[:,:,k+1]) for j in 1:N ]

        # compute the weights (previous weights div by gammakp1)
        weights  = bif.p[k+1].w ./ gammakp1
        # > normalise
        weights /= sum(weights)
        # > resample
        (bif.p[k], ek) = resample(Particles(samples, weights),
                                    essthresh, resampling)
    end

    # combination
    pss = ParticleSet(N, lg.dimx, K)
    pss.p[1] = deepcopy(bif.p[1])

    propS  = Symmetric(inv(Qi + AQiA + BRiB))
    cpropS = chol(propS)
    for k = 2:K-1
        # observation
        yk = observations[:, k]
        # preparing indices for the forward part
        randmult_fwd = rand(Multinomial(N, psf.p[k-1].w))
        randmult_bwd = rand(Multinomial(N, bif.p[k].w))
        # ^ note bif.p[k] and not k+1 since kth weight is scaled by γ_t+1
        # unroll indices into masks
        indices_fwd = [i for k in 1:N for i in ones(Int,randmult_fwd[k])*k]
        indices_bwd = [i for k in 1:N for i in ones(Int,randmult_bwd[k])*k]
        # # get samples in one shot
        noise = cpropS' * randn(lg.dimx, N)
        mus = zeros(lg.dimx, N)
        for j = 1:N
            mus[:, j] = QiA' * bif.p[k+1].x[indices_bwd[j]] +
                          QiA * psf.p[k-1].x[indices_fwd[j]] + RiB' * yk
        end
        mus = propS * mus + noise
        xk  = [mus[:, j] for j in 1:N]
        # no resampling since perfect sampling
        pss.p[k] = Particles(xk, ones(N)/N)
    end
    pss.p[K] = deepcopy(psf.p[K])
    pss
end

# """
#     particle_fbpf
#
# Forward Backward Particle Filter
# """
# function particlesmoother_fbpf(hmm::HMM, observations::Matrix{Float},
#                                N::Int, M::Int,
#                                propfwd::Proposal, propbwd::Proposal,
#                                npasses::Int=3 ;
#                                resampling::Function=multinomialresampling,
#                                essthresh::Float=0.5
#                                )::Tuple{ParticleSet,Vector{Float}}
#     # PASS 1 = particle filter
#     (pf, ess) = particlefilter(hmm, observations, N, propfwd)
#     # PASS 2 = filter using PF as guide
#     (ps, ess) = particlesmoother_llbbis(hmm, observations, pf, M, propfwd)
#     # PASS 3 = filter using PS as guide etc...
#     K = length(pf)
#
#     pss = ParticleSet(N, hmm.dimx, K)
#     ess = zeros(K)
#
#     (p1,e1) = resample(
#                 Particles( [proposal.noise() for i in 1:N], ones(N)/N),
#                 essthresh )
#
#     # store
#     pss.p[1] = p1
#     ess[1]   = e1
#
#     for k=2:K-1
#         pkp1  = ps.p[k+1]
#         obsk  = observations[:,k]
#         logak = zeros(N)
#         xk    = similar(pkp1.x)
#         # sample from multinomial
#         r     = rand(Multinomial())
#
#     end
#
# end
