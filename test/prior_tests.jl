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
    priordist = NamedProductPrior(d)
    lp = logprob(priordist, [0.5,0.5])
    @test lp .≈ sum(map(logpdf, d, [0.5,0.5]))
end

@testset "bijector" begin
    rng = Random.MersenneTwister(1234)
    # univariate prior
    d = Normal(0,1)
    priordist = prior(:x, d)
    x1 = rand(rng, priordist)
    b1 = bijector(priordist)
    z1 = b1(x1)
    @test z1[1] ≈ x1[1]
    # multivariate prior
    d = (x = Normal(0,1), p = Beta(1,1))
    priordist = NamedProductPrior(d)
    x2 = rand(rng, priordist)
    b2 = bijector(priordist)
    z2 = b2(x2)
    @test z2[1] ≈ x2[1]
    @test z2[2] ≈ Bijectors.Logit(0,1)(x2[2])
    @test all(x2 .≈ inverse(b2)(z2))
    # test bijector stacking
    bs = SBI.bstack(Stacked(b1), b2)
    x3 = vcat(x1, x2)
    z3 = @inferred bs(x3) # check type stability
    @test all(x3 .≈ inverse(bs)(z3))
end

@testset "Gaussian approximation" begin
    rng = Random.MersenneTwister(1234)
    # check special case of LogNormal and LogitNormal where
    # the transform is exact
    priordist = prior(a = LogNormal(0,1), b=LogitNormal(0,1))
    uprior = gaussian_approx(LaplaceMethod(), priordist; rng)
    @test iszero(mean(uprior))
    @test cov(uprior) == I
end
