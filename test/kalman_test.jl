using SMC, Base.Test

include("_legacy_.jl")

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
