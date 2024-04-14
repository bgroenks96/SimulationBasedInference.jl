abstract type AbstractPrior end

# prior interface methods

"""
    prior(args...; kwargs...)

Generic constructor for prior distribution types that can be implemented
by subtypes of `AbstractPrior`.
"""
function prior end

logprob(prior::AbstractPrior, x) = error("logprob not implemented for prior of type $(typeof(prior))")
logprob(prior::AbstractPrior, x::AbstractMatrix) = sum(map(xᵢ -> logprob(prior, xᵢ), eachcol(x)))

"""
    forward_map(prior::AbstractPrior, x)

Applies the forward map from the sample space of the prior to the parameter space of the
forward model (simulator). Note that this mapping need not necessarily be bijective in the
case of hierarchical or reparamterized formulations of the model parameter prior.
Defaults to returning `identity(x)`.
"""
forward_map(::AbstractPrior, x) = identity(x)

function forward_map(prior::AbstractPrior)
    function(ζ)
        forward_map(prior, ζ)
    end
end

"""
    unconstrained_forward_map(prior::AbstractPrior, ζ)

Applies the forward map from unconstrained to forward model parameter `(g ∘ f): Ζ ↦ Θ ↦ Φ`
where `f⁻¹` is the inverse bijector of `prior` (i.e. mapping from unconstrained Ζ to constrained Θ space)
and `g` is defined by `forward_map` which maps from Θ to the parameter space of the forward model, Φ.
"""
function unconstrained_forward_map(prior::AbstractPrior, ζ)
    f⁻¹ = inverse(bijector(prior))
    g = forward_map(prior)
    return g(f⁻¹(ζ))
end

function unconstrained_forward_map(prior::AbstractPrior)
    function(ζ)
        unconstrained_forward_map(prior, ζ)
    end
end

# External dispatches

Base.names(prior::AbstractPrior) = error("names not implemented")

StatsBase.sample(prior::AbstractPrior, args...; kwargs...) = sample(Random.default_rng(), prior, args...; kwargs...)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior; kwargs...) = rand(rng, prior)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, n::Integer; kwargs...) = rand(rng, prior, n)
StatsBase.sample(rng::AbstractRNG, prior::AbstractPrior, d::Dims; kwargs...) = reshape(reduce(hcat, rand(rng, prior, prod(d))), :, d...)

Base.rand(rng::AbstractRNG, prior::AbstractPrior) = error("rand not implemented for $(typeof(prior))")
Base.rand(rng::AbstractRNG, prior::AbstractPrior, n::Integer) = [rand(rng, prior) for i in 1:n]
Base.rand(prior::AbstractPrior) = rand(Random.default_rng(), prior)
Base.rand(prior::AbstractPrior, n::Integer) = rand(Random.default_rng(), prior, n)

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

logprob(prior::PriorDistribution, x::AbstractVector) = sum(map((dᵢ, xᵢ) -> logpdf(dᵢ, xᵢ), collect(prior.dist), x))

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
