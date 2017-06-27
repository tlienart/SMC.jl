export  KalmanFilter,
        kalmanfilter,
        kalmansmoother

struct KalmanFilter
    means::Matrix{Float}
    covariances::Array{Float,3}
    # Storage for recycling in kalman smoother
    means_::Matrix{Float}
    covariances_::Array{Float,3}
end
struct KalmanSmoother
    means::Matrix{Float}
    covariances::Array{Float,3}
end

function kalmanfilter(lg::LinearGaussian, observations::Matrix{Float},
                      mu0::Vector{Float}, C0::Matrix{Float}
                      )::KalmanFilter
    @assert issymmetric(C0) && isposdef(C0) "Cov mat must be sym + pos def"

    nsteps = size(observations, 2)

    kf_means = zeros(lg.dimx, nsteps)
    kf_covs  = zeros(lg.dimx, lg.dimx, nsteps)

    kf_means[:,1]  = mu0
    kf_covs[:,:,1] = C0

    kf_means_ = copy(kf_means)
    kf_covs_  = copy(kf_covs)

    for k = 2:nsteps
        # intermediate (k|k-1)
        mu_ = lg.A * kf_means[:,k-1]
        C_  = lg.A * kf_covs[:,:,k-1] * lg.A' + lg.Q
        K_  = (C_ * lg.B') / (lg.R + lg.B*C_*lg.B')
        # update (k|k)
        kf_means[:,k]  = mu_ + K_*(observations[:,k]-lg.B*mu_)
        kf_covs[:,:,k] = (eye(lg.dimx) - K_*lg.B)*C_
        # storage for smoothing
        kf_means_[:,k]  = mu_
        kf_covs_[:,:,k] = C_
    end
    KalmanFilter(kf_means, kf_covs, kf_means_, kf_covs_)
end

function kalmansmoother(lg::LinearGaussian, observations::Matrix{Float},
                        kf::KalmanFilter)::KalmanSmoother
    ks_means = similar(kf.means)
    ks_covs  = similar(kf.covariances)
    # Pre-computations
    iQA, iRB = lg.Q\lg.A, lg.R\lg.B
    Pbi   = lg.B'*iRB
    Pbi_K = copy(Pbi)
    c     = iRB'*observations[:,end]
    # Initialisation
    ks_covs[:,:,end] = inv(inv(kf.covariances_[:,:,end]) + Pbi_K )
    ks_means[:,end]  = ks_covs[:,:,end] * (
                        kf.covariances_[:,:,end] \ kf.means_[:,end] + c )
    # Kalman Smoother with 2 Filter Smoother update
    for k   = (size(observations,2)-1):-1:1
        K   = (eye(lg.dimx) + lg.Q*Pbi)\lg.A
        Pbi = Pbi_K+iQA'*(lg.A-K)
        c   = iRB'*observations[:,k] + K'*c
        #
        ks_covs[:,:,k] = inv(inv(kf.covariances_[:,:,k]) + Pbi )
        ks_means[:,k]  = ks_covs[:,:,k] * (
                            kf.covariances_[:,:,k] \ kf.means_[:,k] + c )
    end
    KalmanSmoother(ks_means, ks_covs)
end
