"""
    Likelihood{obsType<:SimulatorObservable}

Base type for specifying "likelihoods", i.e. the sampling distribution
or density describing the data given an `SimulatorObservable`.
"""
abstract type Likelihood{obsType<:SimulatorObservable} end

Base.nameof(l::Likelihood) = l.name

observable(lik::Likelihood) = lik.obs

(lik::Likelihood)(args...) = lik(args...)

struct MvGaussianLikelihood{Tprior,obsType<:SimulatorObservable} <: Likelihood{obsType}
    name::Symbol
    obs::obsType
    prior::Tprior # prior for likelihood parameters (e.g. noise scale)
    MvGaussianLikelihood(name::Symbol, obs::SimulatorObservable, σ_prior=Exponential(1.0)) = new{typeof(σ_prior),typeof(obs)}(name, obs, σ_prior)
end

const IsotropicGaussianLikelihood = MvGaussianLikelihood{<:UnivariateDistribution}
const DiagonalGaussianLikelihood = MvGaussianLikelihood{<:MultivariateDistribution}
const FullGaussianLikelihood = MvGaussianLikelihood{<:MatrixDistribution}

function (lik::MvGaussianLikelihood)(σ)
    μ = vec(retrieve(lik.obs))
    Σ = covariance(lik, σ)
    return MvNormal(μ, Σ)
end

covariance(lik::IsotropicGaussianLikelihood, σ::Number) = Diagonal(σ^2*ones(prod(size(lik.obs))))
covariance(::DiagonalGaussianLikelihood, σ::AbstractVector) = Diagonal(σ.^2)
covariance(::FullGaussianLikelihood, Σ::AbstractMatrix) = Σ
