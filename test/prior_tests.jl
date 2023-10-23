using SimulationBasedInference

using Random
using Test

@testset "Priors" begin
    rng = Random.MersenneTwister(1234)
    priordist = PriorDistribution(Normal(0,1))
    @test isa(rand(rng, priordist), Float64)
    samples = sample(rng, priordist, 100)
    @test isa(samples, Vector)
    priordist = PriorDistribution(MvNormal(zeros(2),1))
    @test isa(rand(priordist), Vector)
    samples = sample(rng, priordist, 100)
    @test isa(samples, Vector{<:Vector})
end
