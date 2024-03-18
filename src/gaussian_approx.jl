abstract type GaussianApproximationMethod end

Base.@kwdef struct EmpiricalGaussian <: GaussianApproximationMethod
    n_samples::Integer = 10_000
    full_cov::Bool = false
end

"""
    gaussian_gapprox(prior::AbstractPrior; num_prior_samples::Int=10_000, rng::Random.AbstractRNG=Random.default_rng())

Builds an empirical multivariate Gaussian approximation of the given `prior` distribution by computing the moments
of the transformed samples.
"""
function gaussian_approx(approx::EmpiricalGaussian, prior::AbstractPrior; rng::Random.AbstractRNG=Random.default_rng())
    constrained_to_unconstrained = bijector(prior)
    constrained_prior_samples = sample(rng, prior, approx.n_samples)
    unconstrained_prior_samples = reduce(hcat, map(constrained_to_unconstrained, constrained_prior_samples))
    unconstrained_mean = mean(unconstrained_prior_samples, dims=2)
    unconstrained_cov = if approx.full_cov
        cov(unconstrained_prior_samples, dims=2)
    else
        Diagonal(var(unconstrained_prior_samples, dims=2)[:,1])
    end
    return MvNormal(unconstrained_mean[:,1], unconstrained_cov)
end

Base.@kwdef struct Laplace <: GaussianApproximationMethod
    roundoff::Int = 8 # number of decimal places to round the mode
end

function gaussian_approx(approx::Laplace, prior::AbstractPrior; rng::Random.AbstractRNG=Random.default_rng())
    function nll(z)
        finv = inverse(bijector(prior))
        x = finv(z)
        -logprob(prior, x)
    end
    constrained_to_unconstrained = bijector(prior)
    z0 = constrained_to_unconstrained(rand(rng, prior))
    result = optimize(nll, z0, BFGS())
    mode = result.minimizer
    # compute Fisher information
    Γ = ForwardDiff.hessian(nll, mode)
    # get covariance
    Σ = pinv(Γ)
    # constructor MvNormal prior
    return MvNormal(round.(mode, digits=approx.roundoff), Σ)
end
