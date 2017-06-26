export kalmanfilter

function kalmanfilter(lg::LinearGaussian, observations::Matrix{Float},
                      mu0::Vector{Float}, C0::Matrix{Float}
                      )::Tuple{Matrix{Float},Array{Float,3}}
    @assert issymmetric(C0) && isposdef(C0) "Cov mat must be sym + pos def"

    T = size(observations, 2)

    kf_mus  = zeros(lg.dimx, T)
    kf_covs = zeros(lg.dimx, lg.dimx, T)

    kf_mus[:,1]    = mu0
    kf_covs[:,:,1] = C0

    Q = lg.cholQ'*lg.cholQ
    R = lg.cholR'*lg.cholR

    for t = 2:T
        # intermediate (t|t-1)
        mu_ = lg.A * kf_mus[:,t-1]
        C_  = lg.A * kf_covs[:,:,t-1] * lg.A' + Q
        K_  = (C_ * lg.B') / (R + lg.B*C_*lg.B')
        # update (t|t)
        kf_mus[:,t]    = mu_ + K_*(observations[:,t]-lg.B*mu_)
        kf_covs[:,:,t] = (eye(lg.dimx) - K_*lg.B)*C_
        # storage for smoothing (?)
        # XXX
    end
    return(kf_mus, kf_covs)
end
