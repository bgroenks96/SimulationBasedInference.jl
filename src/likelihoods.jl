"""
    Likelihood{obsType<:SimulatorObservable}

Base type for specifying "likelihoods", i.e. the sampling distribution
or density describing the data given an `SimulatorObservable`.
"""
abstract type Likelihood{obsType<:SimulatorObservable} end

Base.nameof(l::Likelihood) = l.name

observable(lik::Likelihood)::SimulatorObservable = lik.obs

getprior(lik::Likelihood)::AbstractPrior = lik.prior

(lik::Likelihood)(args...) = lik(args...)

struct MvGaussianLikelihood{priorType<:AbstractPrior,obsType<:SimulatorObservable} <: Likelihood{obsType}
    name::Symbol
    obs::obsType
    prior::priorType # prior for likelihood parameters (e.g. noise scale)
    function MvGaussianLikelihood(name::Symbol, obs::SimulatorObservable, noise_scale_prior::AbstractPrior)
        return new{typeof(noise_scale_prior),typeof(obs)}(name, obs, noise_scale_prior)
    end
end

const IsotropicGaussianLikelihood = MvGaussianLikelihood{<:UnivariatePriorDistribution}
const DiagonalGaussianLikelihood = MvGaussianLikelihood{<:MultivariatePriorDistribution}

function (lik::MvGaussianLikelihood)(σ)
    μ = vec(retrieve(lik.obs))
    Σ = covariance(lik, σ)
    return MvNormal(μ, Σ)
end

covariance(lik::IsotropicGaussianLikelihood, σ::Number) = Diagonal(σ^2*ones(prod(size(lik.obs))))
covariance(lik::IsotropicGaussianLikelihood, σ::AbstractVector) = Diagonal(σ[1]^2*ones(prod(size(lik.obs))))
covariance(::DiagonalGaussianLikelihood, σ::AbstractVector) = Diagonal(σ.^2)

# Joint prior

"""
    JointPrior{priorType<:AbstractPrior,likTypes<:Tuple{Vararg{Likelihood}}} <: AbstractPrior

Represents the prior w.r.t the full joint distribution, `p(x,θ)`, i.e. `p(θ)` where `θ = [θₘ,θₗ]`;
θₘ are the simulator model parameters and θₗ are the observation noise model parameters.
"""
struct JointPrior{priorType<:AbstractPrior,likTypes<:Tuple{Vararg{Likelihood}}} <: AbstractPrior
    model::priorType
    likelihoods::likTypes
end

JointPrior(modelprior::AbstractPrior, liks::Likelihood...) = JointPrior(modelprior, liks)

Base.names(jp::JointPrior) = merge(
    (model=names(jp.model),),
    map(l -> names(getprior(l)), with_names(jp.likelihoods))
)

function Base.rand(rng::AbstractRNG, jp::JointPrior)
    param_nt = merge(
        (model=rand(rng, jp.model),),
        map(l -> rand(rng, getprior(l)), with_names(jp.likelihoods)),
    )
    return ComponentVector(param_nt)
end

function Bijectors.bijector(jp::JointPrior)
    bs = map(l -> bijector(getprior(l)), jp.likelihoods)
    Stacked([bijector(jp.model), bs...])
end
