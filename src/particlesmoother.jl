export
    particlesmoother_ffbs

"""
    particlesmoother_ffbs

http://www.stats.ox.ac.uk/~doucet/briers_doucet_maskell_smoothingstatespacemodels.pdf
Check equation 11
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
        # denominators precomputation: O(N^2) comp, O(N) storage
        ds = [ sum( pk.w[l]*exp(hmm.transloglik(k,pk.x[l],pkp1.x[j]))
                        for l in 1:N ) for j in 1:N ]
        # FFBS formula, in place computations, O(N) for each i so O(N^2) all
        for i in 1:N
            pk.w[i] *= sum( pkp1.w[j] *
                                exp(hmm.transloglik(k,pk.x[i],pkp1.x[j])) /
                                    ds[j] for j in 1:N )
        end
        pk.w /= sum(pk.w)
        # store the updated weights
        psw.p[k].w = copy(pk.w)
    end
    psw
end
