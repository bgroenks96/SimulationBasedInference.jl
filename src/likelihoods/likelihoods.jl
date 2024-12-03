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
forward simulation. When the `SimulatorLikelihood` is evaluated, these outputs are obtained from `getvalue(obs)`
and the only additional parameters needed are those specified by `prior`.
"""
struct SimulatorLikelihood{distType,priorType,obsType,dataType}
    name::Symbol
    obs::obsType
    data::dataType
    prior::priorType

    """
        SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}

    Creates a `SimulatorLikelihood` with the given distribution type, observable, data source, and prior distribtuion.
    A custom identifier may also be specified via the `name` argument; by default, the name of the observable is used.
    """
    function SimulatorLikelihood(::Type{distType}, obs, data, prior, name=nameof(obs)) where {distType}
        return new{distType,typeof(prior),typeof(obs),typeof(data)}(name, obs, data, prior)
    end
end

Base.nameof(l::SimulatorLikelihood) = l.name

"""
    observable(lik::SimulatorLikelihood)::SimulatorObservable

Retrieve the `SimulatorObservable` associated with this likelihood.
"""
observable(lik::SimulatorLikelihood)::SimulatorObservable = lik.obs

"""
    getprior(lik::SimulatorLikelihood)

Return the `AbstractSimulatorPrior` distribution associated with the parameters for this likelihood.
Note that this is distinct from the prior for the forward model parameters.
"""
getprior(lik::SimulatorLikelihood) = lik.prior

"""
    predictive_distribution(lik::SimulatorLikelihood, args...)

Builds the predictive distribution of `lik` given the parameters in `args`. This method is mandatory
for all specializations of `SimulatorLikelihood`.
"""
predictive_distribution(lik::SimulatorLikelihood, args...) = error("not implemented")
predictive_distribution(lik::SimulatorLikelihood, p::NamedTuple) = predictive_distribution(lik, p...)

"""
    sample_prediction([rng::AbstractRNG], lik::SimulatorLikelihood, args...)

Samples the conditional predictive distribution `p(y|u)` where `u` is the current value of the likelihood
observable. This method is optional for specializations; the default implementation simply invokes `rand`
on the `predictive_distribution(lik, args...)`.
"""
sample_prediction(rng::AbstractRNG, lik::SimulatorLikelihood, args...) = rand(rng, predictive_distribution(lik, args...))
sample_prediction(lik::SimulatorLikelihood, args...) = sample_prediction(Random.default_rng(), lik, args...)

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

export GPLikelihood
include("gp_likelihood.jl")

export ImplicitLikelihood
include("implicit_likelihood.jl")

export DiracLikelihood
include("dirac_likelihood.jl")

export JointPrior
include("joint_prior.jl")
