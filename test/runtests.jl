using SMC
using Base.Test

@testset "particles"    begin include("particleset_test.jl") end

@testset "hmm"          begin include("hmm_test.jl")         end
