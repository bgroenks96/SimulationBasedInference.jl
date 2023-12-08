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

const SimulatorSciMLForwardProblem = SimulatorForwardProblem{<:SciMLBase.AbstractSciMLProblem}

"""
    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)

Constructs a generic simulator forward problem from the given `AbstractSciMLProblem`; note that this could be any
problem type, e.g. an optimization problem, nonlinear system, quadrature, etc.
"""
function SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, nothing)
end
"""
    SimulatorForwardProblem(f, p0, observables::SimulatorObservable...)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and the given `observables`.
"""
function SimulatorForwardProblem(f, p0, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(SimpleForwardProblem(f, p0), named_observables, nothing)
end
"""
    SimulatorForwardProblem(f, p0)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and a default transient observable.
"""
SimulatorForwardProblem(f, p0) = SimulatorForwardProblem(f, p0, SimulatorObservable(:y, state -> state.result))

"""
    SciMLBase.remaker_of(forward_prob::SimulatorSciMLForwardProblem)

Returns a function which will rebuild a `SimulatorForwardProblem` from its arguments.
The remaker function additionally provides a keyword argument `copy_observables` which,
if `true`, will `deepcopy` the observables to ensure independence. The default setting is `true`.
"""
function SciMLBase.remaker_of(forward_prob::SimulatorSciMLForwardProblem)
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
DiffEqBase.check_prob_alg_pairing(prob::SimulatorSciMLForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)
DiffEqBase.isinplace(prob::SimulatorSciMLForwardProblem) = DiffEqBase.isinplace(prob.prob)

# Overload property methods to forward properties from `prob` field
Base.propertynames(prob::SimulatorSciMLForwardProblem) = (:prob, :observables, :config, propertynames(prob.prob)...)
function Base.getproperty(prob::SimulatorSciMLForwardProblem, sym::Symbol)
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

# Simple forward function wrapper

struct SimpleForwardProblem{funcType,pType}
    f::funcType
    p::pType
end

mutable struct SimpleForwardSolver
    prob::SimpleForwardProblem
    result
end

CommonSolve.init(prob::SimpleForwardProblem; p=prob.p) = SimpleForwardSolver(SimpleForwardProblem(prob.f, p), missing)

CommonSolve.solve!(solver::SimpleForwardSolver) = solver.result = prob.f(prob.p)
