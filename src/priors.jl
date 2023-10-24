abstract type AbstractPrior end

# prior interface methods

logprob(prior::AbstractPrior, x) = error("logprob not implemented for prior of type $(typeof(prior))")

Base.names(prior::AbstractPrior) = error("names not implemented")

StatsBase.sample(prior::AbstractPrior, args...; kwargs...) = sample(Random.GLOBAL_RNG, prior, args...; kwargs...)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, args...; kwargs...) = rand(rng, prior)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, n::Int, args...; kwargs...) = [rand(rng, prior) for i in 1:n]

Base.rand(rng::AbstractRNG, prior::AbstractPrior) = error("rand not implemented for $(typeof(prior))")

ParameterMapping(prior::AbstractPrior) = ParameterMapping(bijector(prior))

# Simple implementation for Distributions.jl types

"""
    PriorDistribution{distTypes<:NamedTuple}

Simple diagonal prior that wraps a `NamedTuple` of distributions from the `Distributions` package.
"""
struct PriorDistribution{distTypes<:NamedTuple} <: AbstractPrior
    dists::distTypes
end

const UnivariatePriorDistribution{T} = PriorDistribution{NamedTuple{x,Tuple{T}}} where {x,T<:UnivariateDistribution}
const MultivariatePriorDistribution = PriorDistribution{<:NamedTuple}

"""
    PriorDistribution(name::Symbol, dist::Distribution)

Alias for `PriorDistribution((name = dist))`.
"""
PriorDistribution(name::Symbol, dist::Distribution) = PriorDistribution((; name => dist))

logprob(prior::PriorDistribution, x) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), prior.dists, x))

Base.names(prior::PriorDistribution) = keys(prior.dists)

Base.rand(rng::AbstractRNG, prior::PriorDistribution) = ComponentVector(map(dᵢ -> rand(rng, dᵢ), prior.dists))

Bijectors.bijector(prior::UnivariatePriorDistribution) = bijector(prior.dists[1])
Bijectors.bijector(prior::MultivariatePriorDistribution) = Stacked(collect(map(bijector, prior.dists)))

# Hotfix for incorrect implementation of bijector for product distribution.
# See: https://github.com/TuringLang/Bijectors.jl/issues/290
function Bijectors.bijector(prod::Product{Continuous})
    D = eltype(prod.v)
    return if Bijectors.has_constant_bijector(D)
        Bijectors.elementwise(bijector(prod.v[1]))
    else
        Bijectors.Stacked(map(bijector, Tuple(prod.v)))
    end
end
