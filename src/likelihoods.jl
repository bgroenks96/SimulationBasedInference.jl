"""
    SimulatorLikelihood{distType,obsType,dataType,priorType}

Represents a simulator-based likelihood function. A `SimulatorLikelihood`
consists of four basic components:

(1) A distribution type, e.g. `Normal`,

(2) A `SimulatorObservable` which represents the parameter forward map characteristic
    to the likelihood function,

(3) A set of `data`, usually a `Vector` or `Matrix`, which matches the structure of the observable,

(4) A prior distribution governing one or more additional parameters required to compute the likelihood.

Due to the typically high cost of evaluating the parameter forward map, `SimulatorLikelihood` effectively decouples
the computation of the likelihood from the simulator via the `SimulatorObservable`, which stores the result of a
forward simulation. When the `SimulatorLikelihood` is evaluated, these outputs are obtained from `retrieve(obs)`
and the only additional parameters needed are those specified by `prior`.
"""
struct SimulatorLikelihood{distType,obsType,dataType,priorType}
    name::Symbol
    obs::obsType
    data::dataType
    prior::priorType
    function SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}
        return new{distType,typeof(obs),typeof(data),typeof(prior)}(name, obs, data, prior)
    end
end

Base.nameof(l::SimulatorLikelihood) = l.name

getobservable(lik::SimulatorLikelihood)::SimulatorObservable = lik.obs

getprior(lik::SimulatorLikelihood)::AbstractPrior = lik.prior

function loglikelihood(lik::SimulatorLikelihood, args...)
    d = predictive_distribution(lik, args...)
    return logdensity(d, lik.data)
end

function predictive_distribution(lik::SimulatorLikelihood{Normal}, σ)
    μ = retrieve(lik.obs)[1]
    return Normal(μ, σ)
end

function predictive_distribution(lik::SimulatorLikelihood{<:MvNormal}, σ)
    μ = vec(retrieve(lik.obs))
    Σ = cov(lik, σ)
    return MvNormal(μ, Σ)
end

Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::Number) = Diagonal(σ^2*ones(prod(size(lik.data))))
Statistics.cov(lik::SimulatorLikelihood{IsoNormal}, σ::AbstractVector) = cov(lik, σ[1])
Statistics.cov(lik::SimulatorLikelihood{DiagNormal}, σ::AbstractVector) = Diagonal(σ.^2)

# implement SciML interface for reconstructing the type with new values
function SciMLBase.remaker_of(lik::SimulatorLikelihood{distType}) where {distType}
    # by default, just use the type name to reconstruct the likelihood with each parameter;
    # additional dispatches can be added for special cases
    remake(; name=lik.name, obs=lik.obs, data=lik.data, prior=lik.prior) = SimulatorLikelihood(distType, obs, data, prior, name)
end

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
    bs = map(d -> bijector(d), jp.lik)
    Stacked([bijector(jp.model), bs...])
end

function logdensity(jp::JointPrior, x::ComponentVector)
    lp_model = logdensity(jp.model, x.model)
    liknames = keys(jp.lik)
    lp_lik = sum(map((d,n) -> logdensity(d, getproperty(x, n)), jp.lik, liknames))
    return lp_model + lp_lik
end