export
    particlesmoother_ffbs,
    particlesmoother_gtfs

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

# """
#   particlesmoother_bis
#
# Particle smoother based on the Backward Information Smoothing algorithm.
# """
# function particlesmoother_bis(hmm::HMM, psf::ParticleSet;
#                               resampling::Function=multinomialresampling,
#                               essthresh::Float=0.5 )::ParticleSet
#     K = length(psf)
#     N = length(psf.p[1])
#     # particle set smoother (storage)
#     psw = deepcopy(psf)
#     #
#     # Particles at last step need be resampled
#     (pK,eK) = resample(psw.p[K], essthresh, resampling)
#
# end
