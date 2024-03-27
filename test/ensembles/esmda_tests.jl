using SimulationBasedInference

using OrdinaryDiffEq
using SciMLBase
using Test

@testset "ensemble_kalman_analysis: sanity check" begin
    n_par = 2
    n_obs = 10
    n_ens = 32
    alpha = 1.0
    R_cov = 1.0
    dummy_ens = randn(n_par, n_ens)
    dummy_obs = randn(n_obs)
    dummy_pred = randn(n_obs, n_par)*dummy_ens
    post_ens = ensemble_kalman_analysis(dummy_ens, dummy_obs, dummy_pred, alpha, R_cov)
    @test all(isfinite.(post_ens))
end

@testset "ensemble_kalman_analysis: evensen_scalar_nonlinear" begin
    x_true = 1.0
    b_true = 0.2
    σ_y = 0.1
    alpha = 1.0
    n_ens = 1000
    x_prior = Normal(0,1)
    # log-normal with mean 0.1 and stddev 0.2
    b_prior = autoprior(0.1, 0.2, lower=0.0, upper=Inf)
    rng = Random.MersenneTwister(1234)
    testprob = evensen_scalar_nonlinear(x_true, b_true; n_obs=100, rng, x_prior, b_prior)
    # model parameter forward map
    param_map = unconstrained_forward_map(testprob.prior.model)
    transform = bijector(testprob.prior.model)
    inverse_transform = inverse(transform)
    # sample initial ensemble from model prior (excluding likelihood parameters)
    prior = reduce(hcat, rand(rng, testprob.prior.model, n_ens))
    unconstrained_prior = reduce(hcat, map(transform, eachcol(prior)))
    y_pred, _ = SimulationBasedInference.ensemble_solve(
        unconstrained_prior,
        testprob.forward_prob,
        EnsembleThreads(),
        nothing,
        param_map,
        iter=1
    )
    y_obs = testprob.likelihoods.y.data
    y_lik = mean(testprob.prior.lik.y)
    unconstrained_posterior = ensemble_kalman_analysis(unconstrained_prior, y_obs, y_pred, alpha, σ_y^2)
    @test all(isfinite.(unconstrained_posterior))
    posterior = reduce(hcat, map(inverse_transform, eachcol(unconstrained_posterior)))
    # check that ensemble error decreased overall
    @test mean(abs.(posterior .- [x_true, b_true])) < mean(abs.(prior .- [x_true, b_true]))
end

@testset "ES-MDA: solver inteface" begin
    rng = Random.MersenneTwister(1234)
    testprob = evensen_scalar_nonlinear(;rng)
    solver = init(testprob, ESMDA())
    test_ensemble_alg_interface(solver)
end

@testset "ES-MDA: evensen_scalar_nonlinear" begin
    x_true = 1.0
    b_true = 0.2
    σ_y = 0.1
    n_ens = 128
    x_prior = Normal(0,1)
    # log-normal with mean 0.1 and stddev 0.2
    # b_prior = autoprior(0.1, 0.2, lower=0.0, upper=Inf)
    b_prior = Bijectors.TransformedDistribution(Normal(log(0.1), 1), Base.Fix1(broadcast, exp))
    rng = Random.MersenneTwister(1234)
    testprob = evensen_scalar_nonlinear(x_true, b_true; n_obs=100, rng, x_prior, b_prior)
    transform = bijector(testprob.prior.model)
    inverse_transform = inverse(transform)
    alg = ESMDA(maxiters=10)
    testsol = solve(testprob, alg, EnsembleThreads(); n_ens, rng)
    unconstrained_posterior = get_ensemble(testsol.result)
    posterior = reduce(hcat, map(inverse_transform, eachcol(unconstrained_posterior)))
    posterior_mean = mean(posterior, dims=2)[:,1]
    @show posterior_mean
    @test abs(posterior_mean[1] - x_true) < 0.1
    @test abs(posterior_mean[2] - b_true) < 0.1
end

@testset "ES-MDA: Linear ODE inversion" begin
    rng = Random.MersenneTwister(1234)
    # linear ODE test case with default parameter settings
    inference_prob = linear_ode(; rng)
    # parameter prior, excluding likelihood parameters
    prior = inference_prob.prior.model
    eks = ESMDA()
    # solve inference problem with EKS
    eks_sol = solve(inference_prob, eks, EnsembleThreads(), n_ens=128, verbose=false, rng=rng)
    # check results
    u_ens = get_ensemble(eks_sol.result)
    constrained_to_unconstrained = bijector(prior)
    posterior_ens = reduce(hcat, map(inverse(constrained_to_unconstrained), eachcol(u_ens)))
    posterior_mean = mean(posterior_ens, dims=2)
    @test abs(posterior_mean[1] - 0.2) < 0.01
end
