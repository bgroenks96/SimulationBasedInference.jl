"""
    SimulatorLikelihood{distType,obsType,dataType,priorType}

Represents a simulator-based likelihood function. A `SimulatorLikelihood`
consists of four basic components:

(1) A distribution type, e.g. `Normal`,

(2) A `SimulatorObservable` which represents the observation operator,

(3) A set of `data`, usually a `Vector` or `Matrix`, which matches the structure of the observable,

(4) A prior distribution governing one or more additional parameters required to compute the likelihood.

Due to the typically high cost of evaluating the parameter forward map, `SimulatorLikelihood` effectively decouples
the computation of the likelihood from the simulator via the `SimulatorObservable`, which stores the result of a
forward simulation. When the `SimulatorLikelihood` is evaluated, these outputs are obtained from `retrieve(obs)`
and the only additional parameters needed are those specified by `prior`.
"""
struct SimulatorLikelihood{distType,priorType,obsType,dataType}
    name::Symbol
    obs::obsType
    data::dataType
    prior::priorType
    function SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}
        return new{distType,typeof(prior),typeof(obs),typeof(data)}(name, obs, data, prior)
    end
end

Base.nameof(l::SimulatorLikelihood) = l.name

getobservable(lik::SimulatorLikelihood)::SimulatorObservable = lik.obs

getprior(lik::SimulatorLikelihood) = lik.prior

"""
    predictive_distribution(lik::SimulatorLikelihood, args...)

Builds the predictive distribution of `lik` given the parameters in `args`.
"""
predictive_distribution(lik::SimulatorLikelihood, args...) = error("not implemented")
predictive_distribution(lik::SimulatorLikelihood, p::NamedTuple) = predictive_distribution(lik, p...)

"""
    loglikelihood(lik::SimulatorLikelihood, args...)

Evaluates the log-lielihood of `lik` on the current observable state by
constructing the `predictive_distribution` and evaluating the `logpdf` of the data.
"""
function loglikelihood(lik::SimulatorLikelihood, args...)
    d = predictive_distribution(lik, args...)
    return logprob(d, lik.data)
end

# implement SciML interface for reconstructing the type with new values
function SciMLBase.remaker_of(lik::SimulatorLikelihood{distType}) where {distType}
    # by default, just use the type name to reconstruct the likelihood with each parameter;
    # additional dispatches can be added for special cases
    remake(; name=lik.name, obs=lik.obs, data=lik.data, prior=lik.prior) = SimulatorLikelihood(distType, obs, data, prior, name)
end

export GaussianLikelihood, IsotropicGaussianLikelihood, DiagonalGaussianLikelihood
include("gaussian_likelihood.jl")

include("gp_likelihood.jl")

include("joint_prior.jl")
