using SMC, Base.Test
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

srand(155)
(psf, ess) = particlefilter(hmm, observations, 100, bootstrapprop(lg))

@test length(psf)==K

pfm  = mean(psf)
pfmm = zeros(dx,K)
for k in 1:K
    pfmm[:,k] = pfm[k]
end

@test norm(pfmm-states)/norm(states) < 0.4

srand(521)
psffbs  = particlesmoother_ffbs(hmm, psf)

@test length(psffbs)==K

psm  = mean(psffbs)
psmm = zeros(dx,K)
for k in 1:K
    psmm[:,k] = psm[k]
end

@test norm(psmm-states)/norm(states) < 0.2

srand(521)
(psbbis, ess) = particlesmoother_bbis(hmm, psf, observations, bootstrapprop(lg))

psm2  = mean(psbbis)
psmm2 = zeros(dx,K)
for k in 1:K
    psmm2[:,k] = psm2[k]
end

@test norm(psmm2-states)/norm(states) < 0.2
