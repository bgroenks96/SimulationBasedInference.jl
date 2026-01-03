#### Forward problems ####

"""
    SimulatorForwardProblem{simType, paramType, seedType, obsTypes, names} <: SciMLBase.AbstractSciMLProblem

Represents a "forward" problem from parameters/initial conditions to output `SimulatorObservable`s.
This type wraps an underlying simulator, a set of `Observable`s, and an optional RNG seed for stochastic
forward models.

    SimulatorForwardProblem(f, p::AbstractVector, observables::SimulatorObservable...)

Creates a forward problem from the given function or callable type `f` with parameters `p` and
the given `observables`.

    SimulatorForwardProblem(f, 0::AbstractVector)

Creates a forward problem from the given function or callable type `f` with parameters `p` and
a default transient observable. Note that this constructor calls `f(p)` in order to determine
the shape of the observable. If `f` is very costly and this is undesirable, it is recommended
to use the explicit constructor.

    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)

Constructs a generic simulator forward problem from the given `AbstractSciMLProblem`; note that this could be any
problem type, e.g. an optimization problem, nonlinear system, quadrature, etc.
"""
struct SimulatorForwardProblem{simType, paramType, seedType, obsTypes, names} <: SciMLBase.AbstractSciMLProblem
    "Simulator function or type"
    simulator::simType

    "Parameters for the forward simulation"
    p::paramType

    "Observables derived from the simulator output"
    observables::NamedTuple{names, obsTypes}

    "Random number generator for stochastic simulators"
    rng_seed::seedType
end

"""
Type alias for `SimulatorForwardProblem`s where the forward map is defined by a SciML problem type.
"""
const SciMLForwardProblem{probType} = SimulatorForwardProblem{probType} where {probType<:AbstractSciMLProblem}

"""
Type alias for `SimulatorForwardProblem`s where the forward map is defined by a SciML differential equations problem type.
"""
const SciMLDiffEqForwardProblem{probType} = SimulatorForwardProblem{probType} where {probType<:AbstractDEProblem}

# Functions

"""
    SimulatorForwardProblem(simulator, p::AbstractVector, observables::SimulatorObservable...; rng_seed)

Constructs a `SimulatorForwardProblem` from the given simulator, parameters `p`, and observables.
"""
function SimulatorForwardProblem(
    simulator,
    p::AbstractVecOrMat,
    observables::SimulatorObservable...;
    rng_seed = nothing
)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(simulator, p, named_observables, rng_seed)
end

"""
    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...; rng_seed)

Constructs a `SimulatorForwardProblem` from the given SciML problem and observables.
"""
function SimulatorForwardProblem(
    prob::SciMLBase.AbstractSciMLProblem,
    observables::SimulatorObservable...;
    p::AbstractVecOrMat = prob.p,
    rng_seed = nothing
)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, p, named_observables, rng_seed)
end

# SciML method dispatches

"""
    SciMLBase.remake(
        forward_prob::SimulatorForwardProblem;
        prob=forward_prob.simulator,
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
    p=forward_prob.p,
    simulator=forward_prob.simulator,
    observables=forward_prob.observables,
    rng_seed=forward_prob.rng_seed,
    copy_observables=true,
    kwargs...
)
    new_observables = copy_observables ? deepcopy(observables) : observables
    return SimulatorForwardProblem(simulator, p, new_observables, rng_seed)
end
function SciMLBase.remake(
    forward_prob::SciMLForwardProblem;
    p=forward_prob.p,
    prob=forward_prob.simulator,
    observables=forward_prob.observables,
    rng_seed=forward_prob.rng_seed,
    copy_observables=true,
    kwargs...
)
    newprob = remake(prob; p, kwargs...)
    new_observables = copy_observables ? deepcopy(observables) : observables
    return SimulatorForwardProblem(newprob, p, new_observables, rng_seed)
end
SciMLBase.remaker_of(forward_prob::SciMLForwardProblem) = (;kwargs...) -> remake(forward_prob; kwargs...)
SciMLBase.isinplace(prob::SciMLForwardProblem) = DiffEqBase.isinplace(prob.prob)
# Overload property methods to forward properties from underlying SciML problem
Base.propertynames(prob::SciMLForwardProblem) = (:simulator, :observables, :rng_seed, propertynames(getfield(prob, :simulator))...)
function Base.getproperty(prob::SciMLForwardProblem, sym::Symbol)
    if sym ∈ (:simulator, :observables, :rng_seed)
        return getfield(prob, sym)
    end
    return getproperty(prob.simulator, sym)
end

function Base.show(io::IO, ::MIME"text/plain", prob::SimulatorForwardProblem)
    println(io, "SimulatorForwardProblem for $(nameof(typeof(prob.simulator))) with $(length(prob.observables)) observables $(keys(prob.observables))")
end

