using SMC, Base.Test

srand(123)
N = 50000
x = randn(N)
w = rand(N); w /= sum(w)

ps1 = Particles(x, w)
ps2 = multinomialresampling(ps1)

mps1 = mean(ps1)
mps2 = mean(ps2)

# mean and variance of resampled set should be similar to initial.
@test isapprox(mean(ps1), mean(ps2), atol=0.1)
@test isapprox(var(ps2.x), sum(ps1.x.^2 .* ps1.w)-mps1^2, atol=0.1)
