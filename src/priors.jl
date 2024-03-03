abstract type AbstractPrior end

# prior interface methods

"""
    prior(args...; kwargs...)

Generic constructor for prior distribution types that can be implemented
by subtypes of `AbstractPrior`.
"""
function prior end

logprob(prior::AbstractPrior, x) = error("logdensity not implemented for prior of type $(typeof(prior))")

Base.names(prior::AbstractPrior) = error("names not implemented")

StatsBase.sample(prior::AbstractPrior, args...; kwargs...) = sample(Random.default_rng(), prior, args...; kwargs...)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, args...; kwargs...) = rand(rng, prior)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, n::Integer, args...; kwargs...) = rand(rng, prior, n)

Base.rand(rng::AbstractRNG, prior::AbstractPrior) = error("rand not implemented for $(typeof(prior))")
Base.rand(rng::AbstractRNG, prior::AbstractPrior, n::Integer) = [rand(rng, prior) for i in 1:n]
Base.rand(prior::AbstractPrior) = rand(Random.default_rng(), prior)
Base.rand(prior::AbstractPrior, n::Integer) = rand(Random.default_rng(), prior, n)

# note that Bijectors.jl maps constrained -> unconstrained, so we need to take the inverse here
ParameterTransform(prior::AbstractPrior) = ParameterTransform(inverse(bijector(prior)))

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

logprob(prior::PriorDistribution, x) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), prior.dist, x))

Base.names(prior::PriorDistribution) = keys(prior.dist)

Base.rand(rng::AbstractRNG, prior::PriorDistribution) = ComponentVector(map(dᵢ -> rand(rng, dᵢ), prior.dist))

# Statistics

Statistics.mean(prior::PriorDistribution) = map(mean, prior.dist)
Statistics.var(prior::PriorDistribution) = map(var, prior.dist)
Statistics.cov(prior::PriorDistribution) = map(cov, prior.dist)

# Bijectors

Bijectors.bijector(prior::UnivariatePriorDistribution) = bijector(prior.dist[1])
Bijectors.bijector(prior::MultivariatePriorDistribution) = Stacked(collect(map(bijector, prior.dist)))
