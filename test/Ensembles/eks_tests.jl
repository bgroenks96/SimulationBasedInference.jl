using SimulationBasedInference
using SimulationBasedInference.Ensembles

using Bijectors
using EnsembleKalmanProcesses
using LinearAlgebra
using OrdinaryDiffEq
using Test

import Random

@testset "EKS: solver inteface" begin
    rng = Random.MersenneTwister(1234)
    testprob = evensen_scalar_nonlinear(;rng)
    solver = init(testprob, EKS())
    test_ensemble_alg_interface(solver)
end

@testset "EKS: Linear ODE inversion" begin
    rng = Random.MersenneTwister(1234)
    # linear ODE test case with default parameter settings
    inference_prob = linear_ode(; rng)
    # parameter prior, excluding likelihood parameters
    prior = inference_prob.prior.model
    eks = EKS()
    # solve inference problem with EKS
    eks_sol = solve(inference_prob, eks, EnsembleThreads(), n_ens=128, verbose=false, rng=rng)
    # check results
    u_ens = get_u_final(eks_sol.result.ekp)
    constrained_to_unconstrained = bijector(prior)
    posterior_ens = reduce(hcat, map(inverse(constrained_to_unconstrained), eachcol(u_ens)))
    posterior_mean = mean(posterior_ens, dims=2)
    @test abs(posterior_mean[1] - 0.2) < 0.01
end

@testset "EKS: evensen_scalar_nonlinear" begin
    x_true = 1.0
    b_true = 0.2
    σ_y = 0.1
    alpha = 1.0
    n_ens = 128
    x_prior = Normal(0,1)
    # log-normal with mean 0.1 and stddev 0.2
    b_prior = Bijectors.TransformedDistribution(Normal(log(0.1), 1), Base.Fix1(broadcast, exp))
    rng = Random.MersenneTwister(1234)
    testprob = evensen_scalar_nonlinear(x_true, b_true; n_obs=100, rng, x_prior, b_prior)
    transform = bijector(testprob.prior.model)
    inverse_transform = inverse(transform)
    testsol = solve(testprob, EKS(), EnsembleThreads(); n_ens, verbose=false)
    unconstrained_posterior = get_ensemble(testsol.result)
    posterior = reduce(hcat, map(inverse_transform, eachcol(unconstrained_posterior)))
    posterior_mean = mean(posterior, dims=2)[:,1]
    @show posterior_mean
    @test abs(posterior_mean[1] - x_true) < 0.1
    @test abs(posterior_mean[2] - b_true) < 0.1
end
