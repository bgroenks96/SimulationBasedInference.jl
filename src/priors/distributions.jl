# Simple implementation for Distributions.jl types

"""
    NamedProductPrior{distTypes<:NamedTuple}

Simple diagonal prior that wraps a `NamedTuple` of distributions from the `Distributions` package.
"""
struct NamedProductPrior{distTypes<:NamedTuple} <: AbstractSimulatorPrior
    dist::distTypes
end

const UnivariatePriorDistribution{T} = NamedProductPrior{NamedTuple{x,Tuple{T}}} where {x,T<:UnivariateDistribution}
const MultivariatePriorDistribution = NamedProductPrior{<:NamedTuple}

"""
    prior(name::Symbol, dist::Distribution)

Alias for `NamedProductPrior((name = dist))`.
"""
prior(name::Symbol, dist::Distribution) = NamedProductPrior((; name => dist))

"""
    prior(; dists...)

Alias for `NamedProductPrior((; dists...))`.
"""
prior(; dists...) = NamedProductPrior((; dists...))

logprob(prior::NamedProductPrior, x) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), collect(prior.dist), x))

Base.names(prior::NamedProductPrior) = keys(prior.dist)

Base.rand(rng::AbstractRNG, prior::NamedProductPrior) = ComponentVector(map(dᵢ -> rand(rng, dᵢ), prior.dist))

function Base.getproperty(prior::NamedProductPrior, name::Symbol)
    if name == :dist
        return getfield(prior, :dist)
    else
        return getproperty(getfield(prior, :dist), name)
    end
end

# Statistics

Statistics.mean(prior::NamedProductPrior) = map(mean, prior.dist)
Statistics.median(prior::NamedProductPrior) = map(median, prior.dist)
Statistics.quantile(prior::NamedProductPrior, q) = map(Base.Fix2(quantile, q), prior.dist)
Statistics.var(prior::NamedProductPrior) = map(var, prior.dist)
Statistics.cov(prior::NamedProductPrior) = map(cov, prior.dist)

# Bijectors

Bijectors.bijector(prior::UnivariatePriorDistribution) = bijector(prior.dist[1])
Bijectors.bijector(prior::MultivariatePriorDistribution) = Stacked(collect(map(bijector, prior.dist)))
