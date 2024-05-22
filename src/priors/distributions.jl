# Simple implementation for Distributions.jl types

"""
    PriorDistribution{distTypes<:NamedTuple}

Simple diagonal prior that wraps a `NamedTuple` of distributions from the `Distributions` package.
"""
struct PriorDistribution{distTypes<:NamedTuple} <: AbstractPrior
    dist::distTypes
end

const UnivariatePriorDistribution{T} = PriorDistribution{NamedTuple{x,Tuple{T}}} where {x,T<:UnivariateDistribution}
const MultivariatePriorDistribution = PriorDistribution{<:NamedTuple}

"""
    prior(name::Symbol, dist::Distribution)

Alias for `PriorDistribution((name = dist))`.
"""
prior(name::Symbol, dist::Distribution) = PriorDistribution((; name => dist))

"""
    prior(; dists...)

Alias for `PriorDistribution((; dists...))`.
"""
prior(; dists...) = PriorDistribution((; dists...))

logprob(prior::PriorDistribution, x) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), collect(prior.dist), x))

Base.names(prior::PriorDistribution) = keys(prior.dist)

Base.rand(rng::AbstractRNG, prior::PriorDistribution) = ComponentVector(map(dᵢ -> rand(rng, dᵢ), prior.dist))

# Statistics

Statistics.mean(prior::PriorDistribution) = map(mean, prior.dist)
Statistics.median(prior::PriorDistribution) = map(median, prior.dist)
Statistics.quantile(prior::PriorDistribution, q) = map(Base.Fix2(quantile, q), prior.dist)
Statistics.var(prior::PriorDistribution) = map(var, prior.dist)
Statistics.cov(prior::PriorDistribution) = map(cov, prior.dist)

# Bijectors

Bijectors.bijector(prior::UnivariatePriorDistribution) = bijector(prior.dist[1])
Bijectors.bijector(prior::MultivariatePriorDistribution) = Stacked(collect(map(bijector, prior.dist)))
