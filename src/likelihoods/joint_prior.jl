# Joint prior
"""
    JointPrior{modelPriorType<:AbstractPrior,likPriorTypes} <: AbstractPrior

Represents the "joint" prior `p(θₘ,θₗ)` where `θ = [θₘ θₗ]` are the full set of parameters in the joint;
distribution `p(x,θ)`. θₘ are the model (simulator) parameters and θₗ are the noise/error model parameters.
"""
struct JointPrior{modelPriorType<:AbstractPrior,likPriorTypes,lnames} <: AbstractPrior
    model::modelPriorType
    lik::NamedTuple{lnames,likPriorTypes}
end

JointPrior(modelprior::AbstractPrior, liks::SimulatorLikelihood...) = JointPrior(modelprior, map(getprior, with_names(liks)))

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
    unstack(x) = [x]
    unstack(x::Stacked) = x.bs
    bs = map(d -> bijector(d), jp.lik)
    return Stacked(unstack(bijector(jp.model))..., bs...)
end

function logprob(jp::JointPrior, x::ComponentVector)
    lp_model = logprob(jp.model, x.model)
    liknames = collect(keys(jp.lik))
    lp_lik = sum(map((d,n) -> logprob(d, getproperty(x, n)), collect(jp.lik), liknames))
    return lp_model + lp_lik
end