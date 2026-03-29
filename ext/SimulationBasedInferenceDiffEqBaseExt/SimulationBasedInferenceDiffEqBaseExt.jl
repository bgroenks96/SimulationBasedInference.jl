module SimulationBasedInferenceDiffEqBaseExt

using SimulationBasedInference
using SimulationBasedInference: DynamicalSolver

using DiffEqBase
using SciMLBase: AbstractODEProblem, AbstractODEIntegrator

import CommonSolve: init, step!, solve!

const SciMLForwardProblem = SimulationBasedInference.SciMLForwardProblem
const ODESolver = DynamicalSolver{integratorType} where {integratorType<:AbstractODEIntegrator}

# DiffEqBase dispatches to make solve/init interface work correctly
DiffEqBase.check_prob_alg_pairing(prob::SciMLForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)

SciMLBase.done(solver::ODESolver) = SciMLBase.done(solver.integrator)
SciMLBase.postamble!(solver::ODESolver) = SciMLBase.postamble!(solver.integrator)

# forwarding property dispatches to nested integrator
Base.propertynames(solver::ODESolver) = (fieldnames(solver)..., propertynames(solver.sim)...)
function Base.getproperty(solver::ODESolver, sym::Symbol)
    if sym ∈ fieldnames(typeof(solver))
        return getfield(solver, sym)
    else
        return getproperty(getfield(solver, :sim), sym)
    end
end
function Base.setproperty!(solver::ODESolver, sym::Symbol, value)
    if sym ∈ fieldnames(typeof(solver))
        return setfield!(solver, sym, value)
    else
        return setproperty!(getfield(solver, :sim), sym, value)
    end
end

# CommonSolve interface

"""
    init(forward_prob::SimulatorForwardProblem{<:AbstractODEProblem}, args...; p=forward_prob.simulator.p, saveat=[], solve_kwargs...)

Initializes a `DynamicalSolver` for the given forward problem and ODE integrator algorithm. Additional keyword arguments are
passed through to the integrator `init` implementation.
"""
function init(
    forward_prob::SimulatorForwardProblem{<:AbstractODEProblem},
    ode_alg::SciMLBase.AbstractDEAlgorithm,
    args...;
    p=forward_prob.p,
    saveat=[],
    save_everystep=false,
    copy_observables=false,
    solve_kwargs...
)
    return init(Simulator(forward_prob.simulator), forward_prob, ode_alg, args...; p, saveat, save_everystep, copy_observables, solve_kwargs...)
end

"""
    solve!(sim::ODESolver, args... ; kwargs...)

Solves the forward problem using the given diffeq algorithm and parameters `p`.
"""
function solve!(solver::ODESolver, args...; kwargs...)
    while !SimulationBasedInference.isdone(solver.sim)
        step!(solver, args...; kwargs...)
    end
    sol = solve!(solver.sim)
    return SimulatorForwardSolution(solver.prob, sol)
end

end
