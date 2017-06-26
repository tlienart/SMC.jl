
function kf_legacy(obs,lin1,lin2,cov1,cov2,T,initMean,initCov)
# Coded: Jan 25 2016 / T. Lienart
# ---------------------------------------------
    dx = size(lin1,1)
    dy = size(lin2,1)
    #
    kfMean = zeros(dx,T)
    kfCov  = zeros(dx,dx,T)
    # init
    kfMean[:,1]  = initMean
    kfCov[:,:,1] = initCov
    #
    kfMean_ = copy(kfMean)
    kfCov_  = copy(kfCov)
    #
    for t=2:T
        # intermediate (t|t-1)
        mu_ = lin1*kfMean[:,t-1]
        C_  = lin1*kfCov[:,:,t-1]*lin1'+cov1
        K_  = (C_*lin2') / (cov2+lin2*C_*lin2')
        # update (t|t)
        kfMean[:,t]  = mu_+K_*(obs[:,t]-lin2*mu_)
        kfCov[:,:,t] = (eye(dx)-K_*lin2)*C_
        # storage for smoothing
        kfMean_[:,t]  = mu_
        kfCov_[:,:,t] = C_
    end
    return(kfMean,kfCov,kfMean_,kfCov_)
end

function ks_legacy(obs,lin1,lin2,cov1,cov2,T,kfMean_,kfCov_)
# Coded: Jan 25 2016 / T. Lienart
# ---------------------------------------------
    dx = size(kfMean_,1)
    dy = size(obs,1)
    #
    ksMean = kfMean_*0.
    ksCov  = kfCov_*0.
    #
    A = lin1
    B = lin2
    C = cov1
    D = cov2
    # Pre-computations
    iCA = C\A
    iDB = D\B
    #
    Pbi_T = B'*iDB
    Pbi   = copy(Pbi_T)
    c     = iDB'*obs[:,T]
    #
    # Initialization
    ksCov[:,:,T] = inv((inv(kfCov_[:,:,T])) + Pbi_T)
    ksMean[:,T]  = ksCov[:,:,T]*(kfCov_[:,:,T]\kfMean_[:,T]+c)
    #
    # KS 2FS update
    for t = (T-1):-1:1
        K   = (eye(dx)+C*Pbi)\A
        Pbi = Pbi_T+iCA'*(A-K)
        c   = iDB'*obs[:,t]+K'*c
        #
        ksCov[:,:,t] = inv(inv(kfCov_[:,:,t])+Pbi)
        ksMean[:,t]  = ksCov[:,:,t]*(kfCov_[:,:,t]\kfMean_[:,t]+c)
    end
    return ksMean,ksCov
end
