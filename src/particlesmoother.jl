export
    particlesmoother_ffbs

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
        # numerator of FFBS formula
        ffbsnum(i,j) = exp(log(pk.w[i])+hmm.transloglik(k,pk.x[i],pkp1.x[j]))
        # denominator of FFBS formula
        ffbsdenom(j) = sum(ffbsnum(s,j) for s in 1:N)
        # ffbs formula
        for i in 1:N
            pk.w[i] = sum( pkp1.w[j]*ffbsnum(i,j)/ffbsdenom(j) for j in 1:N )
        end
        pk.w /= sum(pk.w)
        # store the updated weights
        psw.p[k].w = copy(pk.w)
    end
    psw
end
