abstract type AbstractSimulatorPrior end

# prior interface methods

"""
    prior(args...; kwargs...)

Generic constructor for prior distribution types that can be implemented
by subtypes of `AbstractSimulatorPrior`.
"""
function prior end

logprob(prior::AbstractSimulatorPrior, x) = error("logprob not implemented for prior of type $(typeof(prior))")
logprob(prior::AbstractSimulatorPrior, x::AbstractMatrix) = sum(map(xᵢ -> logprob(prior, xᵢ), eachcol(x)))

"""
    forward_map(prior::AbstractSimulatorPrior, ζ)

Applies the forward map from the sample space of the prior to the parameter space of the
forward model (simulator). Note that this mapping need not necessarily be bijective in the
case of hierarchical or reparamterized formulations of the model parameter prior.
Defaults to returning `identity(ζ)`.
"""
forward_map(::AbstractSimulatorPrior, ζ) = identity(ζ)

function forward_map(prior::AbstractSimulatorPrior)
    function(ζ)
        forward_map(prior, ζ)
    end
end

"""
    unconstrained_forward_map(prior::AbstractSimulatorPrior, ζ)

Applies the forward map from unconstrained to forward model parameter `(g ∘ f): Ζ ↦ Θ ↦ Φ`
where `f⁻¹` is the inverse bijector of `prior` (i.e. mapping from unconstrained Ζ to constrained Θ space)
and `g` is defined by `forward_map` which maps from Θ to the parameter space of the forward model, Φ.
"""
function unconstrained_forward_map(prior::AbstractSimulatorPrior, ζ)
    f⁻¹ = inverse(bijector(prior))
    g = forward_map(prior)
    return g(f⁻¹(ζ))
end

function unconstrained_forward_map(prior::AbstractSimulatorPrior)
    function(ζ)
        unconstrained_forward_map(prior, ζ)
    end
end

# External dispatches

Base.names(prior::AbstractSimulatorPrior) = error("names not implemented")

StatsBase.sample(prior::AbstractSimulatorPrior, args...; kwargs...) = sample(Random.default_rng(), prior, args...; kwargs...)
StatsBase.sample(rng::AbstractRNG, prior::AbstractSimulatorPrior; kwargs...) = rand(rng, prior)
StatsBase.sample(rng::AbstractRNG, prior::AbstractSimulatorPrior, n::Integer; kwargs...) = rand(rng, prior, n)
StatsBase.sample(rng::AbstractRNG, prior::AbstractSimulatorPrior, d::Dims; kwargs...) = reshape(reduce(hcat, rand(rng, prior, prod(d))), :, d...)

Base.rand(rng::AbstractRNG, prior::AbstractSimulatorPrior) = error("rand not implemented for $(typeof(prior))")
Base.rand(rng::AbstractRNG, prior::AbstractSimulatorPrior, n::Integer) = [rand(rng, prior) for i in 1:n]
Base.rand(prior::AbstractSimulatorPrior) = rand(Random.default_rng(), prior)
Base.rand(prior::AbstractSimulatorPrior, n::Integer) = rand(Random.default_rng(), prior, n)

include("distributions.jl")

include("gaussian_approx.jl")
