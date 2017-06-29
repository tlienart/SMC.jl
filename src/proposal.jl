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
