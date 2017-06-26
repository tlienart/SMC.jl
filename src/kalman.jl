export  KalmanFilter,
        kalmanfilter,
        kalmansmoother

struct KalmanFilter
    means::Matrix{Float}
    covariances::Array{Float,3}
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

    T = size(observations, 2)

    kf_means = zeros(lg.dimx, T)
    kf_covs  = zeros(lg.dimx, lg.dimx, T)

    kf_means[:,1]  = mu0
    kf_covs[:,:,1] = C0

    kf_means_ = copy(kf_means)
    kf_covs_  = copy(kf_covs)

    for t = 2:T
        # intermediate (t|t-1)
        mu_ = lg.A * kf_means[:,t-1]
        C_  = lg.A * kf_covs[:,:,t-1] * lg.A' + lg.Q
        K_  = (C_ * lg.B') / (lg.R + lg.B*C_*lg.B')
        # update (t|t)
        kf_means[:,t]  = mu_ + K_*(observations[:,t]-lg.B*mu_)
        kf_covs[:,:,t] = (eye(lg.dimx) - K_*lg.B)*C_
        # storage for smoothing
        kf_means_[:,t]  = mu_
        kf_covs_[:,:,t] = C_
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
    Pbi_T = copy(Pbi)
    c     = iRB'*observations[:,end]
    # Initialisation
    ks_covs[:,:,end] = inv(inv(kf.covariances_[:,:,end]) + Pbi_T )
    ks_means[:,end]  = ks_covs[:,:,end] * (
                        kf.covariances_[:,:,end] \ kf.means_[:,end] + c )
    # Kalman Smoother with 2 Filter Smoother update
    for t = (size(observations,2)-1):-1:1
        K   = (eye(lg.dimx) + lg.Q*Pbi)\lg.A
        Pbi = Pbi_T+iQA'*(lg.A-K)
        c   = iRB'*observations[:,t] + K'*c
        #
        ks_covs[:,:,t] = inv(inv(kf.covariances_[:,:,t]) + Pbi )
        ks_means[:,t]  = ks_covs[:,:,t] * (
                            kf.covariances_[:,:,t] \ kf.means_[:,t] + c )
    end
    KalmanSmoother(ks_means, ks_covs)
end
