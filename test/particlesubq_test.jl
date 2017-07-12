using SMC, Base.Test
srand(125)

## Linear Gaussian

dx, dy = 10, 6
A = randn(dx,dx); A /= 0.9norm(A)
B = randn(dy,dx); B /= 1.1norm(B)
Q = randn(dx,dx); Q *= Q'; Q += 0.5*eye(dx); Q += Q'; Q /= 5
R = randn(dy,dy); R *= R'; R += 0.5*eye(dy); R += R'; R /= 20

lg  = LinearGaussian(A,B,Q,R)
hmm = HMM(lg)
x0  = randn(dx)

K = 50
N = 500
(states, observations) = generate(lg, x0, K)

srand(155)
x00 = x0+randn(dx)/5
@time kf = kalmanfilter(lg, observations, x00, eye(dx))
println("KF    : $(norm(kf.means-states)/norm(states))")
srand(521)
@time ks = kalmansmoother(lg, observations, kf)
println("KS    : $(norm(ks.means-states)/norm(states))")

srand(155)
@time (psf, ess) = particlefilter(hmm, observations, N, bootstrapprop(lg))

@test length(psf)==K

pfm  = mean(psf)
pfmm = zeros(dx,K)
for k in 1:K
    pfmm[:,k] = pfm[k]
end

@test norm(pfmm-states)/norm(states) < 1.2
println("PF    : $(norm(pfmm-states)/norm(states))")

srand(521)
@time (pslbbis, ess) = particlesmoother_lbbis(hmm, observations,
                                              psf, bootstrapprop(lg))

psm4  = mean(pslbbis)
psmm4 = zeros(dx,K)
for k in 1:K
    psmm4[:,k] = psm4[k]
end

@test norm(psmm4-states)/norm(states) < 0.9
println("PSBISL: $(norm(psmm4-states)/norm(states))")

srand(521)
@time (psllbbis, ess) = particlesmoother_llbbis(hmm, observations, psf,
                                                5round(Int,log(N)),
                                                bootstrapprop(lg))

psm5  = mean(psllbbis)
psmm5 = zeros(dx,K)
for k in 1:K
    psmm5[:,k] = psm5[k]
end

@test norm(psmm5-states)/norm(states) < 0.9
println("PSBISLL: $(norm(psmm5-states)/norm(states))")

srand(521)
@time pss_fh = particlesmoother_fearnhead_lg(lg, observations, psf)

psm  = mean(pss_fh)
psmm = zeros(dx,K)
for k in 1:K
    psmm[:,k] = psm[k]
end

@test norm(psmm - states)/norm(states) < 1.5 # quite poor...
println("PS_FH  : $(norm(psmm-states)/norm(states))")
