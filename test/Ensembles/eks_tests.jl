using SimulationBasedInference
using SimulationBasedInference.Ensembles

using Bijectors
using EnsembleKalmanProcesses
using LinearAlgebra
using OrdinaryDiffEq
using Test

import Random

@testset "Linear ODE inversion" begin
    rng = Random.MersenneTwister(1234)
    # set up linear ODE problem with two parameters
    ode_p = ComponentArray(α=0.2)
    odeprob = ODEProblem((u,p,t) -> -p[1]*u, [1.0], (0.0,1.0), ode_p)
    # observable extracts state from integrator
    observable = SimulatorObservable(:obs, integrator -> integrator.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    # specify forward problem
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    # generate "true" solution
    forward_sol = solve(forwardprob, Tsit5())
    @assert forward_sol.sol.retcode == ReturnCode.Default
    true_obs = retrieve(observable)
    # specify priors
    prior = PriorDistribution(α=Beta(1,1))
    noise_scale = 0.01
    noise_scale_prior = PriorDistribution(:σ, Exponential(noise_scale))
    # create noisy data
    noisy_obs = true_obs .+ noise_scale*randn(rng, 10)
    # simple Gaussian likelihood; note that we're cheating a bit here since we know the noise level a priori
    lik = SimulatorLikelihood(IsoNormal, observable, noisy_obs, noise_scale_prior)
    # inference problem
    inferenceprob = SimulatorInferenceProblem(forwardprob, Tsit5(), prior, lik)
    eks = EKS(128, EnsembleThreads(), prior, rng)
    # solve inference problem with EKS
    eks_sol = solve(inferenceprob, eks, verbose=false, rng=rng)
    # check results
    u_ens = get_u_final(eks_sol.inference_result.ekp)
    constrained_to_unconstrained = bijector(prior)
    posterior_ens = reduce(hcat, map(inverse(constrained_to_unconstrained), eachcol(u_ens)))
    posterior_mean = mean(posterior_ens, dims=2)
    @test abs(posterior_mean[1] - 0.2) < 0.01
end
