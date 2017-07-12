export
    Proposal,
    bootstrapprop

struct Proposal
    noise::Function
    mean::Function
    loglik::Function
end

function bootstrapprop(lg::LinearGaussian)
    hmm = HMM(lg)
    n = nothing
    Proposal(
        (k=n)                  -> lg.cholQ' * randn(lg.dimx),
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
