using SimulationBasedInference

using Bijectors
using LinearAlgebra
using Random
using Test

@testset "Sampling" begin
    rng = Random.MersenneTwister(1234)
    priordist = prior(:x, Normal(0,1))
    @test isa(rand(rng, priordist).x, Float64)
    samples = sample(rng, priordist, 100)
    @test isa(samples, Vector{<:ComponentVector})
    priordist = prior(:X, MvNormal(zeros(2), I))
    @test isa(rand(priordist).X, AbstractVector)
    samples = sample(rng, priordist, 100)
    @test isa(samples, Vector{<:ComponentVector})
    priordists = prior(x = Normal(0,1), p=Beta(1,1))
    draw = rand(rng, priordists)
    @test hasproperty(draw, :x)
    @test hasproperty(draw, :p)
end

@testset "logdensity" begin
    d = Normal(0,1)
    priordist = prior(:x, d)
    lp = logprob(priordist, 0.5)
    @test lp .≈ logpdf(d, 0.5)
    d = (x = Normal(0,1), p = Beta(1,1))
    priordist = PriorDistribution(d)
    lp = logprob(priordist, [0.5,0.5])
    @test lp .≈ sum(map(logpdf, d, [0.5,0.5]))
end

@testset "bijector" begin
    rng = Random.MersenneTwister(1234)
    d = Normal(0,1)
    priordist = prior(:x, d)
    d = (x = Normal(0,1), p = Beta(1,1))
    priordist = PriorDistribution(d)
    x = rand(rng, priordist)
    b = bijector(priordist)
    z = b(x)
    @test z[1] ≈ x[1]
    @test z[2] ≈ Bijectors.Logit(0,1)(x[2])
    @test all(x .≈ inverse(b)(b(x)))
end
