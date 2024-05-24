"""
    GaussianApproximationMethod

Base type for Gaussian approximations of arbitrary prior distributions.
"""
abstract type GaussianApproximationMethod end

"""
    EmpiricalGaussian <: GaussianApproximationMethod

Represents a simple Gaussian approximation method which calculates a sample mean and covariance.
"""
Base.@kwdef struct EmpiricalGaussian <: GaussianApproximationMethod
    n_samples::Integer = 10_000
    full_cov::Bool = false
end

"""
    gaussian_gapprox(prior::AbstractSimulatorPrior; num_prior_samples::Int=10_000, rng::Random.AbstractRNG=Random.default_rng())

Builds an empirical multivariate Gaussian approximation of the given `prior` distribution by computing the moments
of the transformed samples.
"""
function gaussian_approx(approx::EmpiricalGaussian, prior::AbstractSimulatorPrior; rng::Random.AbstractRNG=Random.default_rng())
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

Base.@kwdef struct LaplaceMethod <: GaussianApproximationMethod
    roundoff::Int = 8 # number of decimal places to round the mode
end

function gaussian_approx(
    approx::LaplaceMethod,
    prior::AbstractSimulatorPrior,
    x0::Union{Nothing,AbstractVector}=nothing; 
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer=BFGS(),
)
    function nll(z)
        finv = inverse(bijector(prior))
        x = zero(x0) + finv(z)
        -logprob(prior, x) - logabsdetjac(finv, z)
    end
    constrained_to_unconstrained = bijector(prior)
    x0 = isnothing(x0) ? rand(rng, prior) : x0
    z0 = constrained_to_unconstrained(x0)
    result = optimize(nll, z0, optimizer)
    mode = result.minimizer
    # compute Fisher information
    Γ = ForwardDiff.hessian(nll, mode)
    # get covariance
    Σ = pinv(Γ)
    # constructor MvNormal prior
    return MvNormal(round.(mode, digits=approx.roundoff), Σ)
end
