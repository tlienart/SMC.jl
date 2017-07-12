export
    Proposal,
    bootstrapprop

struct Proposal
    mu0::Vector{Float}
    noise::Function
    mean::Function
    loglik::Function
end

function bootstrapprop(g::GaussianHMM, mu0::Vector{Float}=[0.0])
    hmm = HMM(g)
    n = nothing
    Proposal(
        (mu0==[0.0]) ? zeros(g.dimx) : mu0,
        (k=n)                  -> g.cholQ' * randn(g.dimx),
        (k=n,xkm1=n,yk=n)      -> hmm.transmean(k, xkm1),
        (k=n,xkm1=n,yk=n,xk=n) -> hmm.transloglik(k, xkm1, xk)
    )
end

# # reverese bootstrap (reversed dynamic)
# function rbootstrap(lg::LinearGaussian)
#     hmm = HMM(lg)
#     n   = nothing
#     pia = pinv(lg.A)
#     Proposal(
#         (k=n)                  -> pia * lg.cholQ' * randn(lg.dimx)
#         (k=n,xkp1=n,yk=n)      -> pia * xkp1
#         (k=n,xk=n,yk=n,xkp1)   -> hmm.transloglik(k, xk, xkp1)
#     )
# end
