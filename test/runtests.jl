using SMC
using Base.Test

@testset "particles" begin include("particleset_test.jl")  end
@testset "hmm"       begin include("hmm_test.jl")          end
@testset "kalman"    begin include("kalman_test.jl")       end
@testset "pf,ps"     begin include("particlef+s_test.jl")  end
@testset "ps,subq"   begin include("particlesubq_test.jl") end
