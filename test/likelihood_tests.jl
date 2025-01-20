using SimulationBasedInference

using Random
using Test

@testset "Joint Prior" begin
    observable = SimulatorObservable(:test, identity, 0.0, 0.0:1.0, (1,))
    p_prior = prior(:p, LogNormal(0,1))
    noise_scale_prior = prior(:σ, LogNormal(0,1))
    data = randn(MersenneTwister(1234), 10)
    lik = SimulatorLikelihood(IsoNormal, observable, data, noise_scale_prior)
    jp = JointPrior(p_prior, lik)
    # Test sampling
    ζ = rand(MersenneTwister(1234), jp)
    @test length(ζ) == 2
    @test hasproperty(ζ, :model)
    @test hasproperty(ζ, :test)
    @test hasproperty(ζ.model, :p)
    # Test forward map
    θ = @inferred SBI.unconstrained_forward_map(jp, [0.0,0.0])
    @test θ ≈ [1.0,1.0]
    p = @inferred SBI.forward_map(jp, ζ)
    @test p == ζ
    # Test log density evaluation
    lp = @inferred SBI.logprob(jp, θ)
    @test lp ≈ sum(logpdf.(LogNormal(0,1), θ))
end
