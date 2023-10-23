#### Forward problems ####

"""
    SimulatorForwardProblem{probType<:SciMLBase.AbstractDEProblem,obs,TObs} <: SciMLBase.DEProblem

Represents a "forward" problem from parameters/initial conditions to output `SimulatorObservable`s.
"""
struct SimulatorForwardProblem{probType<:DEProblem,obs,obsType,tcType} <: DEProblem
    prob::probType
    observables::NamedTuple{obs,obsType}
    obs_to_prob_time::tcType # function that converts observable to problem time units
end

function SimulatorForwardProblem(prob::DEProblem, observables::SimulatorObservable...; obs_to_prob_time=default_time_converter(prob))
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, obs_to_prob_time)
end

function SciMLBase.remaker_of(forward_prob::SimulatorForwardProblem)
    function remake_forward_prob(;
        prob=forward_prob,
        observables=forward_prob.observables,
        obs_to_prob_time=forward_prob.obs_to_prob_time,
        kwargs...
    )
        return SimulatorForwardProblem(remake(prob; kwargs...), observables, obs_to_prob_time)
    end
end

default_time_converter(prob) = identity

# DiffEqBase dispatches to make solve/init interface work correctly
DiffEqBase.check_prob_alg_pairing(prob::SimulatorForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)
DiffEqBase.isinplace(prob::SimulatorForwardProblem) = DiffEqBase.isinplace(prob.prob)

# Overload property methods to forward properties from `prob` field
Base.propertynames(prob::SimulatorForwardProblem) = (:prob, :observables, :obs_to_prob_time, propertynames(prob.prob)...)
function Base.getproperty(prob::SimulatorForwardProblem, sym::Symbol)
    if sym ∈ (:prob,:observables,:obs_to_prob_time)
        return getfield(prob, sym)
    end
    return getproperty(getfield(prob, :prob), sym)
end

"""
    SimulatorForwardSolution{TSol}

Solution for a `SimulatorForwardProblem` that wraps the underlying `DESolution`.
"""
struct SimulatorForwardSolution{TSol}
    prob::SimulatorForwardProblem
    sol::TSol
end

#### Inverse/inference problems ####

"""
    SimulatorInferenceProblem{names} <: SciMLBase.AbstractOptimizationProblem{false}

Represents a generic simulation-based Bayesian inference problem for finding the posterior distribution over model parameters given some observed data.
"""
struct SimulatorInferenceProblem{priorType<:AbstractPrior,likType,dataType,names} <: SciMLBase.AbstractOptimizationProblem{false}
    forward_prob::SimulatorForwardProblem
    prior::priorType
    param_map::ParameterMapping
    likelihoods::NamedTuple{names,likType}
    data::NamedTuple{names,dataType}
end
function SimulatorInferenceProblem(
    prob::SimulatorForwardProblem,
    prior::AbstractPrior,
    lik_with_data::Pair{<:Likelihood,<:AbstractArray}...;
    param_map=ParameterMapping(prior),
)
    likelihoods = map(first, lik_with_data)
    data = map(last, lik_with_data)
    likelihoods_with_names = with_names(likelihoods)
    data_with_names = (; map(Pair, keys(likelihoods_with_names), data)...)
    SimulatorInferenceProblem(prob, prior, param_map, likelihoods_with_names, data_with_names)
end

function Base.getproperty(prob::SimulatorInferenceProblem, sym::Symbol)
    if sym ∈ (:forward_prob,:prior,:param_map,:likelihoods,:data)
        return getfield(prob, sym)
    end
    return getproperty(getfield(prob, :prob), sym)
end

Base.names(::SimulatorInferenceProblem{TP,TL,TD,names}) where {TP,TL,TD,names} = names
