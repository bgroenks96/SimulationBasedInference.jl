#### Forward problems ####

"""
    SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem

Represents a "forward" problem from parameters/initial conditions to output `SimulatorObservable`s.
This type wraps an underlying `SciMLProblem`, a set of `Observable`s, and an optional, probelm-dependent
configuration type.

    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)

Constructs a generic simulator forward problem from the given `AbstractSciMLProblem`; note that this could be any
problem type, e.g. an optimization problem, nonlinear system, quadrature, etc.

    SimulatorForwardProblem(f, p0::AbstractVector, observables::SimulatorObservable...)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and the given `observables`.

    SimulatorForwardProblem(f, p0::AbstractVector)

Creates a forward problem from the given function or callable type `f` with initial
parameters `p0` and a default transient observable. Note that this constructor calls `f(p0)`
in order to determine the shape of the observable. If `f` is very costly and this is
undesirable, it is recommended to use the explicit constructor.
"""
struct SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem
    prob::probType
    observables::NamedTuple{names,obsType}
    config::configType
end

const SimulatorSciMLForwardProblem{probType} = SimulatorForwardProblem{probType} where {probType<:SciMLBase.AbstractSciMLProblem}

"""
    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)

Constructs a `SimulatorForwardProblem` from the given SciML problem and observables.
"""
function SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, nothing)
end

"""
    SimulatorForwardProblem(f, p0::AbstractVector, observables::SimulatorObservable...)

Constructs a `SimulatorForwardProblem` from the callable/function `f(x)` and observables. The base problem
will be a `SimpleForwardProblem` which wraps `f(x)` and uses `p0` as the default parameter/input values for `x`.
"""
function SimulatorForwardProblem(f, p0::AbstractVector, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(SimpleForwardProblem(f, p0), named_observables, nothing)
end

"""
    SciMLBase.remake(
        forward_prob::SimulatorForwardProblem;
        prob=forward_prob.prob,
        observables=forward_prob.observables,
        config=forward_prob.config,
        copy_observables=true,
        kwargs...
    )

Rebuilds a `SimulatorForwardProblem` from its individual components. If `copy_observables=true`,
then `remake` will `deepcopy` the observables to ensure independence. The default setting is `true`.
"""
function SciMLBase.remake(
    forward_prob::SimulatorForwardProblem;
    prob=forward_prob.prob,
    observables=forward_prob.observables,
    config=forward_prob.config,
    copy_observables=true,
    kwargs...
)
    new_observables = copy_observables ? deepcopy(observables) : observables
    return SimulatorForwardProblem(remake(prob; kwargs...), new_observables, config)
end

SciMLBase.remaker_of(forward_prob::SimulatorForwardProblem) = (;kwargs...) -> remake(forward_prob; kwargs...)

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

function Base.show(io::IO, ::MIME"text/plain", prob::SimulatorForwardProblem)
    println(io, "SimulatorForwardProblem ($(nameof(typeof(prob.prob)))) with $(length(prob.observables)) observables $(keys(prob.observables))")
end

"""
    SimulatorForwardSolution{TSol}

Solution for a `SimulatorForwardProblem` that wraps the underlying `DESolution`.
"""
struct SimulatorForwardSolution{TSol}
    prob::SimulatorForwardProblem
    sol::TSol
end

get_observables(sol::SimulatorForwardSolution) = sol.prob.observables

get_observable(sol::SimulatorForwardSolution, name::Symbol) = getvalue(getproperty(get_observables(sol), name))

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
