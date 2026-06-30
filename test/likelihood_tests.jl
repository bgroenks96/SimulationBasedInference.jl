using SimulationBasedInference
using SimulationBasedInference: getprior, predictive_distribution, loglikelihood, observable

using Distributions
using LinearAlgebra
using Random
using SciMLBase
using Test

# ============================================================================
# Gaussian Likelihood Tests - IsoNormal
# ============================================================================

@testset "GaussianLikelihood - IsoNormal" begin
    # Create simple observable and data
    obs = SimulatorObservable(identity, (10,), name=:y, output=Transient())
    
    # Create simulation data with observed values
    simdata = SimulationData()
    observed_values = randn(10)
    store!(simdata, :y, observed_values)
    
    # Create likelihood
    noise_prior = prior(:σ, Exponential(1.0))
    lik = IsotropicGaussianLikelihood(obs, observed_values, noise_prior)
    
    # Test that we can get the prior
    @test getprior(lik) isa NamedProductPrior
    
    # Test predictive distribution with known parameters
    μ_pred = vec(getvalue(simdata, obs))
    σ_val = 0.5
    pred_dist = predictive_distribution(simdata, lik, (; σ=σ_val))
    
    @test pred_dist isa MvNormal
    @test mean(pred_dist) ≈ μ_pred
    @test cov(pred_dist) ≈ Diagonal(σ_val^2 * ones(10))
    
    # Test loglikelihood evaluation
    lp = loglikelihood(simdata, lik, (; σ=σ_val))
    @test isa(lp, Float64)
    @test isfinite(lp)
end

@testset "GaussianLikelihood - DiagNormal" begin
    obs = SimulatorObservable(identity, (10,), name=:y, output=Transient())
    
    simdata = SimulationData()
    observed_values = randn(10)
    store!(simdata, :y, observed_values)
    
    # Diagonal likelihood with per-observation noise scales
    noise_prior = prior(:σs, Product(fill(LogNormal(-1, 1), 10)))
    lik = DiagonalGaussianLikelihood(obs, observed_values, noise_prior)
    
    @test getprior(lik) isa NamedProductPrior
    
    # Test with vector of noise scales
    σ_vals = rand(Exponential(0.5), 10)
    pred_dist = predictive_distribution(simdata, lik, (; σs=σ_vals))
    
    @test pred_dist isa MvNormal
    @test cov(pred_dist) ≈ Diagonal(σ_vals.^2)
end

@testset "GaussianLikelihood - TimeSampled Observable" begin
    # Test with time-sampled observable (the problematic case from problem_tests)
    tsave = 0.0:0.1:1.0
    obs = SimulatorObservable(
        state -> state,
        (1,),
        name=:y,
        output=TimeSampled(0.0, tsave; samplerate=0.01)
    )
    
    # Create simulation data by "running" a simple process
    simdata = SimulationData()
    
    # Simulate 10 time points with value 1.0 at each save point
    for t in tsave
        store!(simdata, :y, [1.0])
    end
    
    observed_data = ones(length(tsave))
    noise_prior = prior(:σ, Exponential(0.5))
    lik = IsotropicGaussianLikelihood(obs, observed_data, noise_prior)
    
    # Test predictive distribution
    σ_val = 0.2
    pred_dist = predictive_distribution(simdata, lik, (; σ=σ_val))
    
    @test pred_dist isa MvNormal
    @test length(mean(pred_dist)) == length(tsave)
    @test cov(pred_dist) ≈ Diagonal(σ_val^2 * ones(length(tsave)))
end

@testset "GaussianLikelihood - LogProbability Evaluation" begin
    obs = SimulatorObservable(identity, (5,), name=:y, output=Transient())
    
    simdata = SimulationData()
    observed_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    store!(simdata, :y, observed_values)
    
    noise_prior = prior(:σ, Exponential(1.0))
    lik = IsotropicGaussianLikelihood(obs, observed_values, noise_prior)
    
    # Test loglikelihood with different noise scales
    lp_small = loglikelihood(simdata, lik, (; σ=0.1))  # Small noise, should be high if model fits
    lp_large = loglikelihood(simdata, lik, (; σ=10.0))  # Large noise, should be lower
    
    @test isfinite(lp_small)
    @test isfinite(lp_large)
    
    # With perfect fit (mean matches data), small noise should give higher likelihood
    μ_pred = observed_values  # Perfect prediction
    lp_perfect = loglikelihood(simdata, lik, (; σ=0.1))
    @test lp_perfect > -100  # Should be reasonably high
end

@testset "GaussianLikelihood - Multi-dimensional Observables" begin
    # Test with 2D observable (e.g., spatial field)
    obs = SimulatorObservable(
        state -> state,
        (3, 4),  # 3x4 grid
        name=:field,
        output=Transient()
    )
    
    simdata = SimulationData()
    observed_field = randn(3, 4)
    store!(simdata, :field, observed_field)
    
    noise_prior = prior(:σ, Exponential(1.0))
    lik = IsotropicGaussianLikelihood(obs, vec(observed_field), noise_prior)
    
    # Test that we can evaluate likelihood
    σ_val = 0.5
    lp = loglikelihood(simdata, lik, (; σ=σ_val))
    
    @test isfinite(lp)
end

@testset "GaussianLikelihood - Isotropic vs Diagonal" begin
    obs = SimulatorObservable(identity, (10,), name=:y, output=Transient())
    
    simdata = SimulationData()
    observed_values = randn(10)
    store!(simdata, :y, observed_values)
    
    # IsoNormal: single noise parameter for all observations
    iso_lik = IsotropicGaussianLikelihood(obs, observed_values, prior(:σ, Exponential(1.0)))
    
    # DiagNormal: separate noise parameter per observation  
    diag_lik = DiagonalGaussianLikelihood(obs, observed_values, 
        prior(:σs, Product(fill(LogNormal(-1, 1), 10)))
    )
    
    # Both should produce valid likelihoods
    iso_lp = loglikelihood(simdata, iso_lik, (; σ=0.5))
    diag_lp = loglikelihood(simdata, diag_lik, (; σs=fill(0.5, 10)))
    
    @test isfinite(iso_lp)
    @test isfinite(diag_lp)
end

# ============================================================================
# SimulatorLikelihood Base Tests
# ============================================================================

@testset "SimulatorLikelihood Interface" begin
    obs = SimulatorObservable(identity, (5,), name=:y, output=Transient())
    
    simdata = SimulationData()
    observed_values = randn(5)
    store!(simdata, :y, observed_values)
    
    noise_prior = prior(:σ, Exponential(1.0))
    lik = SimulatorLikelihood(IsoNormal, obs, observed_values, noise_prior)
    
    # Test accessor methods
    @test observable(lik) === obs
    @test getprior(lik) === noise_prior
    
    # Test that we can reconstruct with new parameters
    new_lik = remake(lik; data=observed_values .* 2.0)
    @test new_lik.data == observed_values .* 2.0
end

# ============================================================================
# Joint Prior Tests
# ============================================================================

@testset "Joint Prior" begin
    observable = SimulatorObservable(identity, (1,), name = :test, output = TimeSampled(0.0, 0.0:1.0))
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
