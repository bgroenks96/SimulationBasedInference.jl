using SimulationBasedInference

using ComponentArrays
using LinearAlgebra
using Random
using Test

@testset "PriorDistribution" begin
    @testset "Sampling" begin
        rng = Random.MersenneTwister(1234)
        priordist = PriorDistribution(:x, Normal(0,1))
        @test isa(rand(rng, priordist).x, Float64)
        samples = sample(rng, priordist, 100)
        @test isa(samples, Vector{<:ComponentVector})
        priordist = PriorDistribution(:X, MvNormal(zeros(2), I))
        @test isa(rand(priordist).X, AbstractVector)
        samples = sample(rng, priordist, 100)
        @test isa(samples, Vector{<:ComponentVector})
        priordists = PriorDistribution((x = Normal(0,1), p=Beta(1,1)))
        draw = rand(rng, priordists)
        @test hasproperty(draw, :x)
        @test hasproperty(draw, :p)
    end
    @testset "logprob" begin
        d = Normal(0,1)
        priordist = PriorDistribution(:x, d)
        lp = logprob(priordist, 0.5)
        @test lp .≈ logpdf(d, 0.5)
        d = (x = Normal(0,1), p = Beta(1,1))
        priordist = PriorDistribution(d)
        lp = logprob(priordist, [0.5,0.5])
        @test lp .≈ sum(map(logpdf, d, [0.5,0.5]))
    end
end
