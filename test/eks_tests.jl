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
    ode_p = ComponentArray(α=0.5, s=0.1)
    odeprob = ODEProblem((u,p,t) -> -p[1]*u .+ p[2], [1.0], (0.0,1.0), ode_p)
    observable = BufferedObservable(:obs, state -> state.u, 0.0, 0.1:0.1:1.0, samplerate=0.01)
    forwardprob = SimulatorForwardProblem(odeprob, observable)
    forward_sol = solve(forwardprob, Tsit5())
    @assert forward_sol.sol.retcode == ReturnCode.Default
    true_obs = retrieve(observable)[1,:]
    prior = PriorDistribution((α=Beta(1,1), s=Normal(0,1)))
    noise_scale_prior = PriorDistribution(:σ, Exponential(0.01))
    lik = MvGaussianLikelihood(:obs, observable, noise_scale_prior)
    noisy_obs = true_obs .+ 0.01*randn(rng, 10)
    inferenceprob = SimulatorInferenceProblem(forwardprob, Tsit5(), prior, lik => noisy_obs)
    constrained_to_unconstrained = bijector(prior)
    prior_samples = reduce(hcat, map(constrained_to_unconstrained, sample(rng, prior, 1000)))
    unconstrained_mean = mean(prior_samples, dims=2)
    unconstrained_cov = var(prior_samples, dims=2)
    eks_prior = MvNormal(unconstrained_mean[:,1], Diagonal(unconstrained_cov[:,1]))
    eks = EKS(256, EnsembleThreads(), eks_prior)
    eks_sol = solve(inferenceprob, eks, verbose=false, rng=rng)
    u_ens = get_u_final(eks_sol["ekp"])
    posterior_ens = reduce(hcat, map(inverse(constrained_to_unconstrained), eachcol(u_ens)))
    posterior_mean = mean(posterior_ens, dims=2)
    @test abs(posterior_mean[1] - 0.5) < 0.05
    @test abs(posterior_mean[2] - 0.1) < 0.05
end
