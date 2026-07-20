"""
    SimulatorForwardSolution{solType,probType}

Solution for a `SimulatorForwardProblem` that wraps the underlying forward solution.
"""
struct SimulatorForwardSolution{solType,probType<:SimulatorForwardProblem,dataType<:SimulationData}
    "Forward problem"
    prob::probType

    "Solution/output data produced by the simulator"
    sol::solType

    "Simulation data storage"
    simdata::dataType
end

get_observables(sol::SimulatorForwardSolution) = map(obs -> getvalue(sol.simdata, obs), sol.prob.observables)

get_observable(sol::SimulatorForwardSolution, name::Symbol) = getvalue(sol.simdata, getproperty(sol.prob.observables, name))

function init(prob::SimulatorForwardProblem, forward_alg=nothing, args...; kwargs...)
    return init(Simulator(prob.simulator), prob, forward_alg, args...; kwargs...)
end

# Observable ordering

"""
    sourcename(obs::Observable)

Return the name of the source observable that `obs` aggregates, or `nothing` if it samples
the raw simulator state directly.
"""
sourcename(obs::TimeSampledObservable) = obs.output.source
sourcename(::Observable) = nothing

"""
    sort_observables(observables)

Return the observables sorted so that every aggregating (sourced) observable appears after the
observable it depends on, guaranteeing a source is observed before its dependents within a single
solver step. Ordering is by dependency depth (0 for observables that sample raw state,
`depth(source) + 1` for aggregators) via a stable sort, so the declared order is preserved among
independent observables. Errors on unknown source names, sources that are not themselves
`TimeSampled` observables (only those persist a reduced output series), and cyclic dependencies.
"""
function sort_observables(observables)
    by_name = Dict{Symbol,Any}(nameof(obs) => obs for obs in observables)
    function depth(obs, seen=Symbol[])
        src = sourcename(obs)
        isnothing(src) && return 0
        haskey(by_name, src) || error("observable :$(nameof(obs)) references unknown source :$src")
        isa(by_name[src], TimeSampledObservable) || error("source :$src of observable :$(nameof(obs)) is not a TimeSampled observable")
        nameof(obs) in seen && error("cyclic observable dependency detected involving :$(nameof(obs))")
        return 1 + depth(by_name[src], push!(seen, nameof(obs)))
    end
    # compute depths eagerly for every observable so validation and cycle detection always run
    # (Julia's `sort` skips the `by` function for singleton collections)
    depths = Dict{Symbol,Int}(nameof(obs) => depth(obs) for obs in observables)
    return sort(collect(observables); by=(obs -> depths[nameof(obs)]), alg=Base.Sort.MergeSort)
end

# Forward solver types

abstract type ForwardSolver{simType} end

## Simple forward maps

mutable struct ForwardMapSolver{
    simulatorType,
    probType<:SimulatorForwardProblem{simulatorType},
    dataType<:SimulationData,
    argsType,
    kwargsType
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Simulation data storage"
    simdata::dataType

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
    simdata::SimulationData=SimulationData(),
    kwargs...
)
    prob = remake(prob; p)
    return ForwardMapSolver(prob, simdata, args, kwargs)
end

step!(::ForwardMapSolver, args...; kwargs...) = error("step! not defined for non-iterative simulators")

function solve!(solver::ForwardMapSolver)
    output = if isnothing(solver.prob.rng_seed)
        solver.prob.simulator(solver.prob.p, solver.args...; solver.kwargs...)
    else
        solver.prob.simulator(solver.prob.p, solver.args...; seed=solver.prob.rng_seed, solver.kwargs...)
    end
    # compute observables
    for obs in sort_observables(solver.prob.observables)
        initialize!(solver.simdata, obs, output)
        observe!(solver.simdata, obs, output)
    end
    return SimulatorForwardSolution(solver.prob, output, solver.simdata)
end

## Iterative simulations

"""
    IterativeSolverIterativeSolver{simulationType, simulatorType, probType<:SimulatorForwardProblem{simulatorType}}

Forward solver for forward problems of `Iterative` simulators.
"""
mutable struct IterativeSolver{
    simulationType,
    simulatorType,
    probType<:SimulatorForwardProblem{simulatorType},
    dataType<:SimulationData
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Simulation data storage"
    simdata::dataType

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
    simdata::SimulationData=SimulationData(),
    maxiters=1000,
    kwargs...
)
    prob = remake(prob; p)
    sim = if isnothing(prob.rng_seed)
        init(prob.simulator, forward_alg, args...; p, kwargs...)
    else
        init(prob.simulator, forward_alg, args...; seed=prob.rng_seed, p, kwargs...)
    end
    # initialize observables
    for obs in sort_observables(prob.observables)
        initialize!(simdata, obs, sim)
    end
    return IterativeSolver(prob, simdata, sim, 1, maxiters)
end

function step!(solver::IterativeSolver, args...; kwargs...)
    result = step!(solver.sim, args...; kwargs...)
    for obs in sort_observables(solver.prob.observables)
        observe!(solver.simdata, obs, solver.sim)
    end
    solver.iter += 1
    return result
end

function solve!(solver::IterativeSolver, args...; kwargs...)
    while !isdone(solver.sim) && solver.iter <= solver.maxiters
        step!(solver, args...; kwargs...)
    end
    sol = solve!(solver.sim) # construct solution
    return SimulatorForwardSolution(solver.prob, sol, solver.simdata)
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
    probType<:SimulatorForwardProblem{simulatorType},
    dataType<:SimulationData
} <: ForwardSolver{simulatorType}
    "Forward problem that started the simulation"
    prob::probType

    "Simulation data storage"
    simdata::dataType

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
    simdata::SimulationData=SimulationData(),
    kwargs...
)
    prob = remake(prob; p)
    # initialize dynamical simulation
    sim = if isnothing(prob.rng_seed)
        init(prob.simulator, forward_alg, args...; p, kwargs...)
    else
        init(prob.simulator, forward_alg, args...; seed=prob.rng_seed, p, kwargs...)
    end
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
    for obs in sort_observables(prob.observables)
        initialize!(simdata, obs, sim)
    end
    return DynamicalSolver(prob, simdata, sim, t_stops, 1)
end

function step!(solver::DynamicalSolver, args...; kwargs...)
    # extract fields from forward integrator and compute dt
    prob = solver.prob
    sim = solver.sim
    t = solver.iter <= length(solver.tstops) ? solver.tstops[solver.iter] : last(solver.tstops)
    dt = max(zero(t), t - current_time(sim))
    # if there are no more stopping points, just forward to the integrator and return
    if solver.iter > length(solver.tstops)
        return step!(sim)
    end
    # otherwise, evaluate the next step and observables if dt > 0
    retval = nothing
    if dt > zero(dt)
        # if sim is a DEIntegrator, always set `stop_at_tdt` to true
        args = isa(sim, SciMLBase.DEIntegrator) ? (true, args...) : args
        retval = step!(sim, dt, args...; kwargs...)
    end
    # iterate over observables and update those for which t is a sample point
    for obs in sort_observables(prob.observables)
        if t ∈ sampletimes(typeof(t), obs)
            observe!(solver.simdata, obs, sim)
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
    return SimulatorForwardSolution(solver.prob, sol, solver.simdata)
end

# Ensemble forward problems

@enum ValidationResult OK RunAgain Skip Fail

default_validator_func(sol, i) = OK

"""
Alias for `SimulatorForwardProblem` with matrix-valued parameters.
"""
const EnsembleForwardProblem{simType} = SimulatorForwardProblem{simType,paramType} where {paramType<:AbstractMatrix}

function solve(
    prob::SimulatorForwardProblem,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix=prob.p,
    kwargs...
)
    prob = remake(prob; p)
    return solve(prob, nothing, ensalg, args...; kwargs...)
end

function solve(
    prob::SimulatorForwardProblem,
    forward_alg,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix=prob.p,
    kwargs...
)
    prob = remake(prob; p)
    return solve(prob, forward_alg, ensalg, args...; p, kwargs...)
end

struct EnsembleForwardSolver{
    algType<:EnsembleAlgorithm,
    dataType,
    solverType
}
    ensalg::algType
    simdata::dataType
    solvers::Vector{solverType}
end

"""
    solve(
        forward_prob::EnsembleForwardProblem,
        forward_alg,
        ensalg::EnsembleAlgorithm,
        args...;
        prob_func=(prob, i) -> prob,
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
    simdata::Vector{<:SimulationData} = default_ensemble_simdata(p),
    prob_func=(prob, i) -> prob,
    kwargs...
)
    enssolver = init(forward_prob, forward_alg, ensalg, args...; p, simdata, prob_func, kwargs...)
    return solve!(enssolver)
end

function solve(
    forward_prob::EnsembleForwardProblem,
    ensalg::EnsembleAlgorithm,
    args...;
    kwargs...
)
    return solve(forward_prob, nothing, ensalg, args...; kwargs...)
end

function init(
    forward_prob::EnsembleForwardProblem,
    forward_alg,
    ensalg::EnsembleAlgorithm,
    args...;
    p::AbstractMatrix=forward_prob.p,
    simdata::Vector{<:SimulationData} = default_ensemble_simdata(p),
    prob_func=(prob, i) -> prob,
    kwargs...
)
    @assert length(simdata) == size(p, 2) "Number of simdata must match the size of the ensemble"
    forward_probs = [prob_func(remake(forward_prob; p=p_i), i) for (i, p_i) in enumerate(eachcol(p))]
    return init(forward_probs, simdata, ensalg, forward_alg, args...; kwargs...)
end

function default_ensemble_simdata(ps::AbstractMatrix; attrs...)
    dataset = SimulationDataSet()
    return [allocate!(dataset, p; attrs...) for p in eachcol(ps)]
end

# Serial

function init(
    forward_probs::Vector{<:SimulatorForwardProblem},
    simdata::Vector{<:SimulationData},
    ensalg::EnsembleSerial,
    args...;
    kwargs...
)
    solvers = map(forward_probs, simdata) do prob, simdata
        init(prob, args...; simdata, kwargs...)
    end
    return EnsembleForwardSolver(ensalg, simdata, solvers)
end

function step!(enssolver::EnsembleForwardSolver{EnsembleSerial})
    results = map(enssolver.solvers) do solver
        step!(solver)
    end
    return results
end

function solve!(enssolver::EnsembleForwardSolver{EnsembleSerial})
    sols = map(enssolver.solvers) do solver
        solve!(solver)
    end
    return sols
end

# Threads

function init(
    forward_probs::Vector{<:SimulatorForwardProblem},
    simdata::Vector{<:SimulationData},
    ensalg::EnsembleThreads,
    args...;
    kwargs...
)
    solvers = Vector(undef, length(forward_probs))
    Threads.@threads for i in 1:length(forward_probs)
        solvers[i] = init(forward_probs[i], args...; simdata=simdata[i], kwargs...)
    end
    return EnsembleForwardSolver(ensalg, simdata, collect(solvers))
end

function step!(enssolver::EnsembleForwardSolver{EnsembleThreads})
    results = Vector(undef, length(enssolver.solvers))
    Threads.@threads for (i, solver) in collect(enumerate(enssolver.solvers))
        results[i] = step!(solver)
    end
    return collect(results)
end

function solve!(enssolver::EnsembleForwardSolver{EnsembleThreads})
    sols = Vector(undef, length(enssolver.solvers))
    Threads.@threads for i in 1:length(sols)
        sols[i] = solve!(enssolver.solvers[i])
    end
    return collect(sols)
end
