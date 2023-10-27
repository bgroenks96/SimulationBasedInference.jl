"""
    SimulatorForwardSolver{stateType}

Basic forward solver that simply wraps the value returned by `init` for the underlying problem.
"""
struct SimulatorForwardSolver{probType,stateType}
    prob::SimulatorForwardProblem{probType}
    state::stateType
end

"""
    SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)

Constructs a generic simulator forward problem from the given `AbstractSciMLProblem`; note that this could be any
problem type, e.g. an optimization problem, nonlinear system, quadrature, etc.
"""
function SimulatorForwardProblem(prob::SciMLBase.AbstractSciMLProblem, observables::SimulatorObservable...)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, nothing)
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
