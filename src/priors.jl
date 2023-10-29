abstract type AbstractPrior end

# prior interface methods

logdensity(prior::AbstractPrior, x) = error("logdensity not implemented for prior of type $(typeof(prior))")

Base.names(prior::AbstractPrior) = error("names not implemented")

StatsBase.sample(prior::AbstractPrior, args...; kwargs...) = sample(Random.GLOBAL_RNG, prior, args...; kwargs...)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, args...; kwargs...) = rand(rng, prior)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, n::Int, args...; kwargs...) = [rand(rng, prior) for i in 1:n]

Base.rand(rng::AbstractRNG, prior::AbstractPrior) = error("rand not implemented for $(typeof(prior))")

# note that Bijectors.jl maps constrained -> unconstrained, so we need to take the inverse here
ParameterMapping(prior::AbstractPrior) = ParameterMapping(inverse(bijector(prior)))

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
    PriorDistribution(name::Symbol, dist::Distribution)

Alias for `PriorDistribution((name = dist))`.
"""
PriorDistribution(name::Symbol, dist::Distribution) = PriorDistribution((; name => dist))

"""
    PriorDistribution(; dists...)

Alias for `PriorDistribution((; dists...))`.
"""
PriorDistribution(; dists...) = PriorDistribution((; dists...))

logdensity(prior::PriorDistribution, x) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), prior.dist, x))

Base.names(prior::PriorDistribution) = keys(prior.dist)

Base.rand(rng::AbstractRNG, prior::PriorDistribution) = ComponentVector(map(dᵢ -> rand(rng, dᵢ), prior.dist))

# Statistics

Statistics.mean(prior::PriorDistribution) = map(mean, prior.dist)
Statistics.var(prior::PriorDistribution) = map(var, prior.dist)
Statistics.cov(prior::PriorDistribution) = map(cov, prior.dist)

# Bijectors

Bijectors.bijector(prior::UnivariatePriorDistribution) = bijector(prior.dist[1])
Bijectors.bijector(prior::MultivariatePriorDistribution) = Stacked(collect(map(bijector, prior.dist)))
