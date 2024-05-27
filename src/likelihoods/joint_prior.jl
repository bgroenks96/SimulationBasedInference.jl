# Joint prior
"""
    JointPrior{modelPriorType<:AbstractSimulatorPrior,likPriorTypes} <: AbstractSimulatorPrior

Represents the "joint" prior `p(θₘ,θₗ)` where `θ = [θₘ θₗ]` are the full set of parameters in the joint;
distribution `p(x,θ)`. θₘ are the model (simulator) parameters and θₗ are the noise/error model parameters.
"""
struct JointPrior{modelPriorType<:AbstractSimulatorPrior,likPriorTypes,lnames} <: AbstractSimulatorPrior
    model::modelPriorType
    lik::NamedTuple{lnames,likPriorTypes}
    ax::Tuple{Vararg{ComponentArrays.Axis}}
end

"""
Constructs a `JointPrior` from the given prior and likelihoods.
"""
function JointPrior(modelprior::AbstractSimulatorPrior, liks::SimulatorLikelihood...)
    lik_priors = with_names(filter(!isnothing, map(prior, liks)))
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

function forward_map(jp::JointPrior, θ::ComponentVector)
    ϕ_m = forward_map(jp.model, θ.model)
    ϕ_lik = map(n -> n => forward_map(jp.lik[n], θ[n]), keys(jp.lik))
    return ComponentVector(;model=ϕ_m, ϕ_lik...)
end
forward_map(jp::JointPrior, θ::AbstractVector) = forward_map(jp, ComponentVector(θ, jp.ax))

function unconstrained_forward_map(jp::JointPrior, ζ::ComponentVector)
    # get inverse bijectors
    f_m = inverse(bijector(jp.model))
    f_lik = map(inverse ∘ bijector, jp.lik)
    # apply bijections
    θ_m = f_m(ζ.model)
    θ_lik = ComponentVector(; map(n -> n => f_lik[n](ζ[n]), keys(jp.lik))...)
    # apply forward maps
    ϕ_m = forward_map(jp.model, θ_m)
    ϕ_lik = map(n -> n => forward_map(jp.lik[n], θ_lik[n]), keys(jp.lik))
    return ComponentVector(;model=ϕ_m, ϕ_lik...)
end
unconstrained_forward_map(jp::JointPrior, θ::AbstractVector) = unconstrained_forward_map(jp, ComponentVector(θ, jp.ax))

bstack(b1, b2) = Stacked(b1, b2)
function bstack(b1::Stacked, b2)
    last_in = last(b1.ranges_in[end])+1
    last_out = last(b1.ranges_out[end])+1
    ranges_in = vcat(b1.ranges_in, [last_in:last_in])
    ranges_out = vcat(b1.ranges_out, [last_out:last_out])
    length_in = b1.length_in + 1
    length_out = b1.length_out + 1
    return Stacked(vcat(b1.bs, b2), ranges_in, ranges_out, length_in, length_out)
end
function bstack(b1, b2::Stacked)
    ranges_in = vcat([1:1], map(r -> first(r)+1:last(r)+1, b2.ranges_in))
    ranges_out = vcat([1:1], map(r -> first(r)+1:last(r)+1, b2.ranges_out))
    length_in = b1.length_in + 1
    length_out = b1.length_out + 1
    return Stacked(vcat(b1.bs, b2), ranges_in, ranges_out, length_in, length_out)
end
function bstack(b1::Stacked, b2::Stacked)
    offs_in = last(b1.ranges_in[end])
    offs_out = last(b1.ranges_out[end])
    return Stacked(
        vcat(b1.bs, b2.bs),
        vcat(b1.ranges_in, map(r -> first(r)+offs_in:last(r)+offs_in, b2.ranges_in)),
        vcat(b1.ranges_out, map(r -> first(r)+offs_out:last(r)+offs_out, b2.ranges_out)),
        b1.length_in + b2.length_in,
        b1.length_out + b2.length_out,
    )
end
