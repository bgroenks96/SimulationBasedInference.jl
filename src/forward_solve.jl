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

function init(prob::SimulatorForwardProblem, forward_alg = nothing, args...; kwargs...)
    return init(Simulator(prob.simulator), prob, forward_alg, args...; kwargs...)
end

# Forward solver types

abstract type ForwardSolver{simType} end

## Simple forward maps

mutable struct ForwardMapSolver{
    simulatorType,
    probType<:SimulatorForwardProblem{simulatorType},
    argsType,
    kwargsType
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Positional arguments for the simulator function"
    args::argsType

    "Keyword arguments for the simulator function"
    kwargs::kwargsType
end

function init(
    ::ForwardMap,
    prob::SimulatorForwardProblem,
    ::Nothing,
    args...;
    p=prob.p,
    copy_observables=false,
    kwargs...
)
    newprob = remake(prob; p, copy_observables)
    return ForwardMapSolver(newprob, args, kwargs)
end

step!(::ForwardMapSolver, args...; kwargs...) = error("step! not defined for non-iterative simulators")

function solve!(sim::ForwardMapSolver)
    output = sim.prob.simulator(sim.prob.p, sim.args...; sim.kwargs...)
    # compute observables
    for obs in sim.prob.observables
        initialize!(obs, output)
        observe!(obs, output)
    end
    return SimulatorForwardSolution(sim.prob, output)
end

## Iterative simulations

"""
    IterativeSolverIterativeSolver{simulationType, simulatorType, probType<:SimulatorForwardProblem{simulatorType}}

Forward solver for forward problems of `Iterative` simulators.
"""
mutable struct IterativeSolver{
    simulationType,
    simulatorType,
    probType<:SimulatorForwardProblem{simulatorType}
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Simulation object"
    sim::simulationType

    "Iteration number"
    iter::Int

    "Maximum number of iterations"
    maxiters::Int
end

function init(
    ::Iterative,
    prob::SimulatorForwardProblem,
    forward_alg,
    args...;
    p=prob.p,
    copy_observables=false,
    maxiters=1000,
    kwargs...
)
    newprob = remake(prob; p, copy_observables)
    sim = if isnothing(prob.rng_seed)
        init(newprob.simulator, forward_alg, args...; kwargs...)
    else
        init(newprob.simulator, forward_alg, args...; seed = prob.rng_seed, kwargs...)
    end
    # initialize observables
    for obs in newprob.observables
        initialize!(obs, sim)
    end
    return IterativeSolver(newprob, sim, 1, maxiters)
end

function step!(solver::IterativeSolver, args...; kwargs...)
    result = step!(solver.sim, args...; kwargs...)
    for obs in solver.prob.observables
        observe!(obs, solver.sim)
    end
    solver.iter += 1
    return result
end

function solve!(solver::IterativeSolver, args...; kwargs...)
    while !isdone(solver.sim) && solver.iter <= solver.maxiters
        step!(solver, args...; kwargs...)
    end
    sol = solve!(solver.sim) # construct solution
    return SimulatorForwardSolution(solver.prob, sol)
end

## Dynamical system simulations

"""
    DynamicalSolver{
        simulationType,
        simulatorType,
        timeType,
        probType<:SimulatorForwardProblem{simulatorType}
    } <: ForwardSolver{simulatorType}

Solver type for forward problems of `Dynamical` simulators.
"""
mutable struct DynamicalSolver{
    simulationType,
    simulatorType,
    timeType,
    probType<:SimulatorForwardProblem{simulatorType}
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Simulation object"
    sim::simulationType

    "Stopping times"
    tstops::Vector{timeType}

    "Iteration number"
    iter::Int
end

function init(
    ::Dynamical,
    prob::SimulatorForwardProblem,
    forward_alg,
    args...;
    p=prob.p,
    copy_observables=false,
    kwargs...
)
    newprob = remake(prob; p, copy_observables)
    # initialize dynamical simulation
    sim = init(newprob.simulator, forward_alg, args...; kwargs...)
    t = current_time(sim)
    tspan = timespan(sim)
    ttype = typeof(t)
    # collect and combine sample points from all TimeSampled observables
    t_sample = map(obs -> sampletimes(ttype, obs), prob.observables)
    t_sample_all = sort(unique(union(t_sample...)))
    t_stops = if isempty(t_sample_all) || t_sample_all[end] < tspan[end]
        vcat(t_sample_all, [ttype(tspan[end])])
    else
        t_sample_all
    end
    # initialize observables
    for obs in newprob.observables
        initialize!(obs, sim)
    end
    return DynamicalSolver(newprob, sim, t_stops, 1)
end

function step!(solver::DynamicalSolver, args...; kwargs...)
    # extract fields from forward integrator and compute dt
    prob = solver.prob
    sim = solver.sim
    t = solver.tstops[solver.iter]
    dt = t - current_time(sim)
    # if there are no more stopping points, just forward to the integrator and return
    if solver.iter > length(solver.tstops)
        return step!(sim)
    end
    # otherwise, evaluate the next step and observables if dt > 0
    retval = nothing
    if dt > zero(dt)
        # if sim is a DEIntegrator, always set `stop_at_tdt` to true
        args = isa(sim, SciMLBase.DEIntegrator) ? (true, args...) : args
        step!(sim, dt, args...; kwargs...)
    end
    # iterate over observables and update those for which t is a sample point
    for obs in prob.observables
        if t ∈ sampletimes(typeof(t), obs)
            observe!(obs, sim)
        end
    end
    # increment step index
    solver.iter += 1
    return retval
end

function solve!(solver::DynamicalSolver, args...; kwargs...)
    while !isdone(solver.sim) && current_time(solver.sim) < maximum(timespan(solver.sim))
        step!(solver, args...; kwargs...)
    end
    sol = solve!(solver.sim)
    return SimulatorForwardSolution(solver.prob, sol)
end

# Ensemble forward problems

"""
Alias for `SimulatorForwardProblem` with matrix-valued parameters.
"""
const EnsembleForwardProblem{simType} = SimulatorForwardProblem{simType, paramType} where {paramType<:AbstractMatrix}

function solve(
    prob::SimulatorForwardProblem,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix = forward_prob.p,
    kwargs...
)
    newprob = remake(prob; p)
    return solve(newprob, nothing, ensalg, args...; kwargs...)
end

function solve(
    prob::SimulatorForwardProblem,
    forward_alg,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix = forward_prob.p,
    kwargs...
)
    newprob = remake(prob; p)
    return solve(newprob, forward_alg, ensalg, args...; kwargs...)
end

function solve(
    forward_prob::EnsembleForwardProblem,
    ensalg::EnsembleAlgorithm,
    args...;
    kwargs...
)
    return solve(forward_prob, nothing, ensalg, args...; kwargs...)
end


"""
    solve(
        forward_prob::EnsembleForwardProblem,
        forward_alg,
        ensalg::EnsembleAlgorithm,
        args...;
        prob_func=(prob, i, repeat) -> remake(prob, p=prob.p[:,i]),
        output_func=ensemble_output_func(forward_prob),
        kwargs...
    )

Solve an `EnsembleProblem` based on the given `SimulatorForwardProblem` and
ensemble algorithm. By default, the parameter ensemble is assumed to be the second
dimension of the parameter matrix in `forward_prob`.
"""
function solve(
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
    return solve(ensprob, forward_alg, ensalg, args...; trajectories=size(p, ensdim), kwargs...)
end

function ensemble_output_func(::SimulatorForwardProblem; validator = (sol, i) -> OK)
    function output(sol::SimulatorForwardSolution, i)
        result = validator(sol, i)
        if result == OK
            # retrieve observables and flatten into a vector
            observables = map(obs -> getvalue(obs), sol.prob.observables)
            return (; sol, observables), false
        elseif result == RunAgain
            observables = nothing
            return (; sol, observables), true
        elseif result == Fail
            error("forward solution validation failed: $sol")
        end
    end
end
