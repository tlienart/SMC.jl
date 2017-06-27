using SMC, Base.Test

srand(123)
N  = 50000
ps = randn(N)
ws = rand(N); ws /= sum(ws)

ps1 = ParticleSet(ps, ws)
ps2 = multinomialresampling(ps1)

mps1 = mean(ps1)
mps2 = mean(ps2)

# mean and variance of resampled set should be similar to initial.
@test isapprox(mean(ps1), mean(ps2), atol=0.1)
@test isapprox(var(ps2.p), sum(ps1.p.^2 .* ps1.w)-mps1^2, atol=0.1)
