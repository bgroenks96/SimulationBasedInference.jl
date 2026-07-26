abstract type AbstractLikelihood end

"""
    SimulatorLikelihood{distType,obsType,dataType,priorType}

Represents a simulator-based likelihood function. A `SimulatorLikelihood`
consists of four basic components:

(1) A distribution type, e.g. `Normal`,

(2) A `SimulatorObservable` which represents the observation operator,

(3) A set of `data`, usually a `Vector` or `Matrix`, which matches the structure of the observable,

(4) A prior distribution governing one or more additional parameters required to compute the likelihood.

Due to the typically high cost of evaluating the parameter forward map, `SimulatorLikelihood` effectively decouples
the computation of the likelihood from the simulator via the `SimulatorObservable`. When the `SimulatorLikelihood`
is evaluated, the observable output is obtained from the simulation data via `getvalue(data, obs)` and the only
additional parameters needed are those specified by `prior`.
"""
struct SimulatorLikelihood{distType,priorType,obsType,dataType} <: AbstractLikelihood
    name::Symbol
    obs::obsType
    data::dataType
    prior::priorType
end

"""
    SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}

Creates a `SimulatorLikelihood` with the given distribution type, observable, data source, and prior distribtuion.
A custom identifier may also be specified via the `name` argument; by default, the name of the observable is used.
"""
function SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}
    return SimulatorLikelihood{distType,typeof(prior),typeof(obs),typeof(data)}(name, obs, data, prior)
end

Base.nameof(l::SimulatorLikelihood) = l.name

"""
    observable(lik::SimulatorLikelihood)

Retrieve the `SimulatorObservable` associated with this likelihood.
"""
observable(lik::SimulatorLikelihood) = lik.obs

"""
    getprior(lik::SimulatorLikelihood)

Return the `AbstractSimulatorPrior` distribution associated with the parameters for this likelihood.
Note that this is distinct from the prior for the forward model parameters.
"""
getprior(lik::SimulatorLikelihood) = lik.prior

"""
    predictive_distribution(data::SimulationData, lik::SimulatorLikelihood, args...)

Builds the predictive distribution of `lik` given the simulation `data` (from which the
observable value is read) and the parameters in `args`. This method is mandatory for all
specializations of `SimulatorLikelihood`. The likelihood is stateless and is evaluated
per-simulation; the `SimulationData` is passed as the first argument.
"""
predictive_distribution(::SimulationData, lik::SimulatorLikelihood, args...) = error("not implemented")
predictive_distribution(data::SimulationData, lik::SimulatorLikelihood, p::NamedTuple) = predictive_distribution(data, lik, p...)

"""
    sample_prediction([rng::AbstractRNG], data::SimulationData, lik::SimulatorLikelihood, args...)

Samples the conditional predictive distribution `p(y|u)` where `u` is the observable value for
the given simulation `data`. This method is optional for specializations; the default
implementation simply invokes `rand` on the `predictive_distribution(data, lik, args...)`.
"""
sample_prediction(rng::AbstractRNG, data::SimulationData, lik::SimulatorLikelihood, args...) = rand(rng, predictive_distribution(data, lik, args...))
sample_prediction(data::SimulationData, lik::SimulatorLikelihood, args...) = sample_prediction(Random.default_rng(), data, lik, args...)

"""
    loglikelihood(data::SimulationData, lik::SimulatorLikelihood, args...)

Evaluates the log-likelihood of `lik` on the observable value stored in `data` by
constructing the `predictive_distribution` and evaluating the `logpdf` of the data.
"""
function loglikelihood(data::SimulationData, lik::SimulatorLikelihood, args...)
    d = predictive_distribution(data, lik, args...)
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

export ImplicitLikelihood
include("implicit_likelihood.jl")

export DiracLikelihood
include("dirac_likelihood.jl")

export JointPrior
include("joint_prior.jl")
