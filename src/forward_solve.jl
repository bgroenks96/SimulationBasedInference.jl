"""
    SimulatorForwardSolution{solType,probType}

Solution for a `SimulatorForwardProblem` that wraps the underlying forward solution.
"""
struct SimulatorForwardSolution{solType, probType<:SimulatorForwardProblem}
    "Forward problem"
    prob::probType

    "Solution/output data produced by the simulator"
    sol::solType
end

get_observables(sol::SimulatorForwardSolution) = sol.prob.observables

get_observable(sol::SimulatorForwardSolution, name::Symbol) = getvalue(getproperty(get_observables(sol), name))

init(prob::SimulatorForwardProblem, args...; kwargs...) = init(prob, SimulatorKind(prob.simulator), args...; kwargs...)

# Simulation types

abstract type Simulation{simType} end

## Function map simulations

mutable struct FunctionMapSimulation{simType, argsType, kwargsType} <: Simulation{simType}
    "Forward problem that started the simulation"
    prob::SimulatorForwardProblem{simType}

    "Positional arguments for the simulator function"
    args::argsType

    "Keyword argumetns for the simulator function"
    kwargs::kwargsType
end

function init(
    prob::SimulatorForwardProblem,
    ::FunctionMap,
    args...;
    p=prob.p,
    copy_observables=false,
    kwargs...
)
    newprob = remake(prob; p, copy_observables)
    return FunctionMapSimulation(newprob, args..., kwargs)
end

step!(::FunctionMapSimulation, args...; kwargs...) = error("step! not defined for non-iterative simulators")

function solve!(sim::FunctionMapSimulation)
    args, kwargs = sim.state
    output = sim.prob.simulator(sim.prob.p, args...; kwargs...)
    return SimulatorForwardSolution(sim.prob, output)
end

## Iterative simulations

"""
    IterativeSimulation{stateType}

Forward solver for `SimulatorForwardProblem`s that holds the original problem type as well
as the simulator state.
"""
mutable struct IterativeSimulation{simType, stateType} <: Simulation{simType}
    "Forward problem that started the simulation"
    prob::SimulatorForwardProblem{simType}

    "Simulator state"
    state::stateType

    "Iteration number"
    iter::Int
end

function init(
    prob::SimulatorForwardProblem,
    ::IterativeSolve,
    args...;
    p=prob.p,
    copy_observables=false,
    kwargs...
)
    newprob = remake(prob; p, copy_observables)
    state = if isnothing(prob.rng_seed)
        init(newprob.simulator, args...; kwargs...)
    else
        init(newprob.simulator, args...; seed = prob.rng_seed, kwargs...)
    end
    # initialize observables
    for obs in newprob.observables
        initialize!(obs, state)
    end
    return IterativeSimulation(newprob, state, 1)
end

function step!(sim::IterativeSimulation, args...; kwargs...)
    result = step!(sim.state, args...; kwargs...)
    for obs in sim.prob.observables
        observe!(obs, sim.state)
    end
    sim.iter += 1
    return result
end

function solve!(sim::IterativeSimulation, args...; kwargs...)
    sol = solve!(sim.state, args...; kwargs...)
    return SimulatorForwardSolution(sim.prob, sol)
end

## Dynamical system simulations

mutable struct DynamicalSimulation{simType, timeType, integratorType} <: Simulation{simType}
    "Forward problem that started the simulation"
    prob::SimulatorForwardProblem{simType}

    "Integrator state"
    integrator::integratorType

    "Stopping times"
    tstops::Vector{timeType}

    "Iteration number"
    iter::Int
end

function init(
    prob::SimulatorForwardProblem,
    ::DynamicalSolve{timeType},
    args...;
    p=prob.p,
    copy_observables=false,
    kwargs...
) where {timeType}
    # collect and combine sample points from all obsevables
    t_sample = map(obs -> sampletimes(timeType, obs), prob.observables)
    t_sample_all = sort(unique(union(t_sample...)))
    t_stops = if isempty(t_sample_all) || t_sample_all[end] < prob.tspan[end]
        vcat(t_sample_all, [timeType(forwprobard_prob.tspan[end])])
    else
        t_sample_all
    end
    newprob = remake(prob; p, copy_observables)
    # initialize integrator with built-in saving disabled
    integrator = init(newprob.simulator, args...; kwargs...)
    # initialize observables
    for obs in newprob.observables
        initialize!(obs, integrator)
    end
    # iterate over observables and update those for which t0 is a sample point
    for obs in newprob.observables
        if integrator.t ∈ sampletimes(timeType, obs)
            observe!(obs, integrator)
        end
    end
    return DynamicalSimulation(newprob, integrator, t_stops, 1)
end

function step!(sim::DynamicalSimulation, args...; kwargs...)
    # extract fields from forward integrator and compute dt
    prob = sim.prob
    integrator = sim.integrator
    t = sim.tstops[sim.iter]
    dt = t - integrator.t
    # if there are no more stopping points, just forward to the integrator and return
    if sim.iter > length(sim.tstops)
        return step!(sim.integrator)
    end
    # otherwise, evaluate the next step and observables if dt > 0
    retval = if dt > zero(dt)
        step!(integrator, dt, true, args...; kwargs...)
    else
        nothing
    end
    # iterate over observables and update those for which t is a sample point
    for obs in prob.observables
        if t ∈ sampletimes(typeof(t), obs)
            observe!(obs, integrator)
        end
    end
    # increment step index
    sim.iter += 1
    return retval
end

function solve!(sim::DynamicalSimulation, args...; kwargs...)
    while !done(sim.integrator)
        step!(sim, args...; kwargs...)
    end
    sol = solve!(sim.integrator)
    return SimulatorForwardSolution(sim.prob, sol)
end

# Ensemble forward problems

"""
Alias for `SimulatorForwardProblem` with matrix-valued parameters.
"""
const EnsembleForwardProblem{simType} = SimulatorForwardProblem{simType, paramType} where {paramType<:AbstractMatrix}

# Special dispatch for ensemble forward problems where no forward algorithm is specified.
function init(
    forward_prob::EnsembleForwardProblem,
    ensalg::EnsembleAlgorithm,
    args...;
    kwargs...
)
    return init(forward_prob, nothing, ensalg, args...; kwargs...)
end

"""
    init(
        forward_prob::EnsembleForwardProblem,
        forward_alg,
        ensalg::EnsembleAlgorithm,
        args...;
        prob_func=(prob, i, repeat) -> remake(prob, p=prob.p[:,i]),
        output_func=ensemble_output_func(forward_prob),
        kwargs...
    )

Initialize an `EnsembleProblem` based on the given `SimulatorForwardProblem` and
ensemble algorithm. By default, the parameter ensemble is assumed to be the second
dimension of the parameter matrix in `forward_prob`.
"""
function init(
    forward_prob::EnsembleForwardProblem,
    forward_alg,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix=forward_prob.p,
    ensdim::Int=2,
    prob_func::Function=(prob, i, repeat) -> remake(prob, p=p[:,i]),
    output_func::Function=ensemble_output_func(forward_prob),
    kwargs...
)
    ensprob = EnsembleProblem(forward_prob; prob_func, output_func)
    # initialize EnsembleProblem
    ens = init(ensprob, forward_alg, ensalg, args...; trajectories=size(Θ, ensdim), kwargs...)
    return ens
end

function ensemble_output_func(::SimulatorForwardProblem; validator = (sol, i) -> OK)
    function output(sol::SimulatorForwardSolution, i)
        result = validator(sol, i)
        if result == OK
            # retrieve observables and flatten into a vector
            observables = reduce(vcat, map(obs -> vec(getvalue(obs)), sol.prob.observables))
            return (; sol, observables), false
        elseif result == RunAgain
            observables = nothing
            return (; sol, observables), true
        elseif result == Fail
            error("forward solution validation failed: $sol")
        end
    end
end
