using SimulationBasedInference.Ensembles
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
    n_obs = 10
    n_ens = 128
    alpha = 1.0
    σ_y = 0.1
    rng = Random.MersenneTwister(1234)
    testcase = evensen_scalar_nonlinear(x_true, b_true; n_obs, n_ens, rng)
    transform = testcase.transform
    inverse_transform = inverse(testcase.transform)
    y_pred = testcase.prior_pred
    y_obs = testcase.obs
    prior = testcase.prior
    prior_mean = mean(prior, dims=2)[:,1]
    unconstrained_prior = reduce(hcat, map(transform, eachcol(prior)))
    σ_y = testcase.hyperparams.σ_y
    unconstrained_posterior = ensemble_kalman_analysis(unconstrained_prior, y_obs, y_pred, alpha, σ_y^2)
    @test all(isfinite.(unconstrained_posterior))
    unconstrained_posterior_mean = mean(unconstrained_posterior, dims=2)[:,1]
    posterior_mean = inverse_transform(unconstrained_posterior_mean)
    prior_mean_resid = prior_mean .- [x_true, b_true]
    posterior_mean_resid = posterior_mean .- [x_true, b_true]
    @test abs(posterior_mean_resid[1]) < abs(prior_mean_resid[1])
end
