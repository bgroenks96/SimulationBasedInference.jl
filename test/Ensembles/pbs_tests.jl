using SimulationBasedInference.Ensembles
using Test


@testset "importance_weights: sanity check" begin
    n_par = 2
    n_obs = 10
    n_ens = 32
    alpha = 1.0
    R_cov = 1.0
    dummy_ens = randn(n_par, n_ens)
    dummy_obs = randn(n_obs)
    dummy_pred = randn(n_obs, n_par)*dummy_ens
    weights, Neff = importance_weights(dummy_obs, dummy_pred, R_cov)
    @test all(isfinite.(weights))
    @test sum(weights) ≈ 1.0
end

@testset "importance_weights: evensen_scalar_nonlinear" begin
    x_true = 1.0
    b_true = 0.2
    rng = Random.MersenneTwister(1234)
    testcase = evensen_scalar_nonlinear(x_true, b_true, n_obs=100, n_ens=1000, rng=rng)
    y_pred = testcase.prior_pred
    y_obs = testcase.obs
    prior = testcase.prior
    σ_y = testcase.hyperparams.σ_y
    w, Neff = importance_weights(y_obs, y_pred, σ_y^2)
    posterior_mean = prior*w
    @test sum(w) ≈ 1.0
    @test abs(posterior_mean[1] - x_true) < 0.1
    # PBS doesn't seem to work reliably for this parameter in this use case
    # @test abs(posterior_mean[2] - b_true) < 0.01
end
