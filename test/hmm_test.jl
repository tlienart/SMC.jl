using SMC, Base.Test

include("_legacy_.jl")

############################

srand(125)

## Linear Gaussian

dx, dy = 5, 3
A = randn(dx,dx); A /= 0.9norm(A)
B = randn(dy,dx); B /= 1.1norm(B)
Q = randn(dx,dx); Q *= Q'; Q += 0.5*eye(dx); Q += Q'; Q /= 5
R = randn(dy,dy); R *= R'; R += 0.5*eye(dy); R += R'; R /= 20

lg  = LinearGaussian(A,B,Q,R)
hmm = HMM(lg)
x0  = randn(dx)

### LinearGaussian -- testing the generation

srand(123)
(states, observations) = generate(lg, x0, 3)

srand(123)
noisex = chol(Q)'*randn(dx, 3)
noisey = chol(R)'*randn(dy, 3)
state1 = x0
state2 = A*state1+noisex[:,2] # we do not use the first noise
state3 = A*state2+noisex[:,3] # since x0 is given
obs1   = B*state1+noisey[:,1]
obs2   = B*state2+noisey[:,2]
obs3   = B*state3+noisey[:,3]

@test state1==states[:,1] &&
      state2==states[:,2] &&
      state3==states[:,3]
@test obs1==observations[:,1] &&
      obs2==observations[:,2] &&
      obs3==observations[:,3]

@test isapprox( hmm.transmean(0,state1), A*state1)
@test isapprox( hmm.obsmean(0,state1), B*state1)
@test isapprox( hmm.transloglik(0,state1,state2),
        (-norm(chol(Q)\(state2-A*state1))^2/2) )
@test isapprox( hmm.obsloglik(0,state2,obs2),
        (-norm(chol(R)\(  obs2-B*state2))^2/2) )

K = 100
(states, observations) = generate(lg, x0, K)

x00 = x0+randn(dx)/5

srand(12)
kf = kalmanfilter(lg, observations, x00, eye(dx))
srand(12)
(kfm_leg, kfc_leg, kfm__leg, kfc__leg) = kf_legacy(observations, A, B, Q,
                                                    R, K, x00, eye(dx))

@test isapprox(kf.means, kfm_leg)
@test isapprox(kf.covs, kfc_leg)
@test isapprox(kf.means_, kfm__leg)
@test isapprox(kf.covs_, kfc__leg)

# fragile test
@test norm(kf.means-states)/norm(states) < 0.25

srand(32)
ks = kalmansmoother(lg, observations, kf)
srand(32)
(ksm_leg, ksc_leg) = ks_legacy(observations, A, B, Q, R, K,
                                kfm__leg, kfc__leg)

@test isapprox(ks.means, ksm_leg)
@test isapprox(ks.covs, ksc_leg)

# fragile test
@test norm(ks.means-states)/norm(states) < 0.2

(ps, ess) = particlefilter(hmm, observations, 100)

@test length(ps)==K

pfm  = mean(ps)
pfmm = zeros(dx,K)
for k in 1:K
    pfmm[:,k] = pfm[k]
end

@test norm(pfmm-states)/norm(states) < 0.4
