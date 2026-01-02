module SimulationBasedInferenceDiffEqBaseExt

using SimulationBasedInference
using SimulationBasedInference: DynamicalSimulation

using DiffEqBase
using SciMLBase: AbstractODEProblem, AbstractODEIntegrator

const SciMLForwardProblem = SimulationBasedInference.SciMLForwardProblem
const ODESimulation{probType} = DynamicalSimulation{probType} where {probType<:AbstractODEIntegrator}

# DiffEqBase dispatches to make solve/init interface work correctly
DiffEqBase.check_prob_alg_pairing(prob::SciMLForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)

SciMLBase.done(solver::ODESimulation) = SciMLBase.done(solver.integrator)
SciMLBase.postamble!(solver::ODESimulation) = SciMLBase.postamble!(solver.integrator)

# forwarding property dispatches to nested integrator
Base.propertynames(sim::ODESimulation) = (fieldnames(sim)..., propertynames(sim.integrator)...)
function Base.getproperty(sim::ODESimulation, sym::Symbol)
    if sym ∈ fieldnames(typeof(sim))
        return getfield(sim, sym)
    else
        return getproperty(getfield(sim, :integrator), sym)
    end
end
function Base.setproperty!(integrator::SimulatorODEForwardSolver, sym::Symbol, value)
    if sym ∈ fieldnames(typeof(integrator))
        return setfield!(integrator, sym, value)
    else
        return setproperty!(getfield(integrator, :integrator), sym, value)
    end
end

# CommonSolve interface

"""
    init(forward_prob::SimulatorForwardProblem{<:AbstractODEProblem}, args...; p=forward_prob.simulator.p, saveat=[], solve_kwargs...)

Initializes a `DynamicalSimulation` for the given forward problem and ODE integrator algorithm. Additional keyword arguments are
passed through to the integrator `init` implementation.
"""
function CommonSolve.init(
    forward_prob::SimulatorForwardProblem{<:AbstractODEProblem},
    args...;
    p=forward_prob.simulator.p,
    saveat=[],
    save_everystep=false,
    copy_observables=false,
    solve_kwargs...
)
    return init(forward_prob, SimulatorKind(forward_prob), args...; p, saveat, save_everystep, copy_observables, solve_kwargs...)
end

"""
    solve!(sim::ODESimulation, stop_at_tdt=true, args... ; kwargs...)

Solves the forward problem using the given diffeq algorithm and parameters `p`.
"""
function CommonSolve.solve!(sim::ODESimulation, stop_at_tdt=true, args...; kwargs...)
    while !done(sim.integrator)
        step!(sim, stop_at_tdt, args...; kwargs...)
    end
    sol = solve!(sim.integrator)
    return SimulatorForwardSolution(sim.prob, sol)
end

end
