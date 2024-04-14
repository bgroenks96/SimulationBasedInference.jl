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
    SimulatorForwardProblem(f, p0::AbstractVector, observables::SimulatorObservable...)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and the given `observables`.
"""
function SimulatorForwardProblem(f, p0::AbstractVector, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(SimpleForwardProblem(f, p0), named_observables, nothing)
end
"""
    SimulatorForwardProblem(f, p0::AbstractVector)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and a default transient observable.
"""
SimulatorForwardProblem(f, p0::AbstractVector) = SimulatorForwardProblem(f, p0, SimulatorObservable(:y, state -> state.u))

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

"""
    SimpleForwardProblem{funcType,pType}

Wrapper type for generic forward models `M` of the form: `y = M(p)` where `p` are the parameters
and `y` are the outputs.
"""
struct SimpleForwardProblem{funcType,pType}
    f::funcType
    p::pType
end

function SciMLBase.remaker_of(prob::SimpleForwardProblem)
    function remake(; f=prob.f, p=prob.p)
        return SimpleForwardProblem(f, p)
    end
end

mutable struct SimpleForwardEval{uType}
    prob::SimpleForwardProblem
    u::uType
end

CommonSolve.init(prob::SimpleForwardProblem, ::Nothing=nothing; p=prob.p) = SimpleForwardEval(SimpleForwardProblem(prob.f, p), prob.f(p))

CommonSolve.solve!(solver::SimpleForwardEval) = solver.u
