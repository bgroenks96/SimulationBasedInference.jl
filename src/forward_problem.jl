#### Forward problems ####

"""
    SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem

Represents a "forward" problem from parameters/initial conditions to output `SimulatorObservable`s.
"""
struct SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem
    prob::probType
    observables::NamedTuple{names,obsType}
    config::configType
end

"""
    SciMLBase.remaker_of(forward_prob::SimulatorForwardProblem)

Returns a function which will rebuild a `SimulatorForwardProblem` from its arguments.
The remaker function additionally provides a keyword argument `copy_observables` which,
if `true`, will `deepcopy` the observables to ensure independence. The default setting is `true`.
"""
function SciMLBase.remaker_of(forward_prob::SimulatorForwardProblem)
    function remake_forward_prob(;
        prob=forward_prob.prob,
        observables=forward_prob.observables,
        config=forward_prob.config,
        copy_observables=true,
        kwargs...
    )
        new_observables = copy_observables ? deepcopy(observables) : observables
        return SimulatorForwardProblem(remake(prob; kwargs...), new_observables, config)
    end
end

# DiffEqBase dispatches to make solve/init interface work correctly
DiffEqBase.check_prob_alg_pairing(prob::SimulatorForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)
DiffEqBase.isinplace(prob::SimulatorForwardProblem) = DiffEqBase.isinplace(prob.prob)

# Overload property methods to forward properties from `prob` field
Base.propertynames(prob::SimulatorForwardProblem) = (:prob, :observables, :config, propertynames(prob.prob)...)
function Base.getproperty(prob::SimulatorForwardProblem, sym::Symbol)
    if sym ∈ (:prob,:observables,:config)
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
