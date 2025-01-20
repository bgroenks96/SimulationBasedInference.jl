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
    ξ = rand(MersenneTwister(1234), jp)
    @test length(ξ) == 2
    @test hasproperty(ξ, :model)
    @test hasproperty(ξ, :test)
    @test hasproperty(ξ.model, :p)
    # Test forward map
    θ = @inferred SBI.unconstrained_forward_map(jp, [0.0,0.0])
    @test θ ≈ [1.0,1.0]
    # Test log density evaluation
    lp = @inferred SBI.logprob(jp, θ)
    @test lp ≈ sum(logpdf.(LogNormal(0,1), θ))
end
