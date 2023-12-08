using Bijectors
using Bijectors:TransformedDistribution
using Distributions
using LinearAlgebra
using SimulationBasedInference
using Random

struct InferenceTestCase{F,T}
    forward_model::F
    transform::T
    hyperparams
    prior
    prior_pred
    obs
end

function evensen_scalar_nonlinear(
    x_true=1.0,
    b_true=0.2;
    n_obs=100,
    n_ens=1000,
    σ_y=0.1,
    x_prior=Normal(0,1),
    b_prior=TransformedDistribution(Normal(log(0.1), 1.0), exp),
    rng=Random.GLOBAL_RNG
)
    function g(θ)
        x = θ[1]
        b = θ[2]
        return x*(1 + b*x^2)
    end
    yt = g([x_true, b_true])
    y = yt .+ σ_y*randn(rng, n_obs)
    x_ens = rand(x_prior, n_ens)
    b_ens = rand(b_prior, n_ens)
    θ_ens = transpose(hcat(x_ens, b_ens))
    y_ens = transpose(repeat(map(g, eachcol(θ_ens)), 1, n_obs))
    hyperparams = (; σ_y)
    transform = bijector(product_distribution([x_prior, b_prior]))
    return InferenceTestCase(g, transform, hyperparams, θ_ens, y_ens, y)
end
