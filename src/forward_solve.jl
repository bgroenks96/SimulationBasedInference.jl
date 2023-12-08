"""
    SimulatorForwardSolver{stateType}

Basic forward solver that simply wraps the value returned by `init` for the underlying problem.
"""
struct SimulatorForwardSolver{probType,stateType}
    prob::SimulatorForwardProblem{probType}
    state::stateType
end

# Generic implementation of `CommonSolve` for any problem type.

function CommonSolve.init(prob::SimulatorForwardProblem, args...; kwargs...)
    state = init(prob.prob, args...; kwargs...)
    # initialize observables
    for obs in prob.observables
        initialize!(obs, state)
    end
    return SimulatorForwardSolver(prob, state)
end

function CommonSolve.step!(solver::SimulatorForwardSolver)
    result = step!(solver.state)
    for obs in solver.prob.observables
        observe!(obs, solver.state)
    end
    return result
end

function CommonSolve.solve!(solver::SimulatorForwardSolver)
    sol = solve!(solver.state)
    for obs in solver.prob.observables
        observe!(obs, solver.state)
    end
    return SimulatorForwardSolution(solver.prob, sol)
end
