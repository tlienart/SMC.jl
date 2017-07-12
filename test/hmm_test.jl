using SMC, Base.Test

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
noisex = chol(Q)' * randn(dx, 3)
noisey = chol(R)' * randn(dy, 3)
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
        (-norm(chol(Q)'\(state2-A*state1))^2/2) )
@test isapprox( hmm.obsloglik(0,obs2,state2),
        (-norm(chol(R)'\(  obs2-B*state2))^2/2) )
