using Test

include("../testcases.jl")

@testset "EKS" begin
    include("eks_tests.jl")
end

@testset "ES-MDA" begin
    include("esmda_tests.jl")
end

@testset "PBS" begin
    include("pbs_tests.jl")
end
