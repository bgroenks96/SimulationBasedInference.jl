# Joint prior
"""
    JointPrior{modelPriorType<:AbstractSimulatorPrior,likNames,likPriorTypes,axesType} <: AbstractSimulatorPrior

Represents the "joint" prior `p(θₘ,θₗ)` where `θ = [θₘ θₗ]` are the full set of parameters in the joint;
distribution `p(x,θ)`. θₘ are the model (simulator) parameters and θₗ are the noise/error model parameters.
"""
struct JointPrior{modelPriorType<:AbstractSimulatorPrior,likNames,likPriorTypes,axesType} <: AbstractSimulatorPrior
    model::modelPriorType
    lik::NamedTuple{likNames,likPriorTypes}
    ax::axesType
end

"""
Constructs a `JointPrior` from the given prior and likelihoods.
"""
function JointPrior(modelprior::AbstractSimulatorPrior, liks::SimulatorLikelihood...)
    lik_priors = (; filter(x -> !isnothing(x[2]), map(lik -> nameof(lik) => getprior(lik), liks))...)
    param_nt = merge(
        (model=rand(modelprior),),
        map(d -> rand(d), lik_priors),
    )
    proto = ComponentVector(param_nt)
    return JointPrior(modelprior, lik_priors, getaxes(proto))
end

Base.names(jp::JointPrior) = merge(
    (model=names(jp.model),),
    map(names, jp.lik),
)

function Base.rand(rng::AbstractRNG, jp::JointPrior)
    param_nt = merge(
        (model=rand(rng, jp.model),),
        map(d -> rand(rng, d), jp.lik),
    )
    return ComponentVector(param_nt)
end

function Bijectors.bijector(jp::JointPrior)
    b_m = bijector(jp.model)
    b_liks = map(d -> bijector(d), jp.lik)
    b_combined = foldl(bstack, tuple(b_m, b_liks...))
    return b_combined
end

function logprob(jp::JointPrior, θ::ComponentVector)
    lp_model = logprob(jp.model, θ.model)
    liknames = collect(keys(jp.lik))
    if length(liknames) > 0
        lp_lik = sum(map((d,n) -> logprob(d, getproperty(θ, n)), collect(jp.lik), liknames))
        return lp_model + lp_lik
    else
        return lp_model
    end
end
logprob(jp::JointPrior, θ::AbstractVector) = logprob(jp, ComponentVector(θ, jp.ax))

@generated function forward_map(jp::JointPrior{<:Any,lnames}, θ::ComponentVector) where {lnames}
    ϕ_args = map(lnames) do n
        :(forward_map(jp.lik[$(QuoteNode(n))], θ[$(QuoteNode(n))]))
    end
    quote
        ϕ_m = forward_map(jp.model, θ.model)
        ϕ = vcat(ϕ_m, $(ϕ_args...))
        return ComponentVector(ϕ, jp.ax)
    end
end
forward_map(jp::JointPrior{<:Any,lnames}, θ::AbstractVector) where {lnames} = forward_map(jp, ComponentVector(θ, jp.ax))

function unconstrained_forward_map(jp::JointPrior, ζ::ComponentVector)
    f = inverse(bijector(jp))
    θ = ComponentArray(f(ζ), jp.ax)
    # apply forward map
    return forward_map(jp.model, θ)
end

function unconstrained_forward_map(jp::JointPrior, θ::AbstractVector)
    return unconstrained_forward_map(jp, ComponentVector(θ, jp.ax))
end
