const DEIntegrator = SciMLBase.DEIntegrator

mutable struct SimulatorForwardIntegrator{algType,uType,tType,integratorType<:DEIntegrator{algType,true,uType,tType}} <: DEIntegrator{algType,true,uType,tType}
    prob::SimulatorForwardProblem
    integrator::integratorType
    tstops::Vector{tType}
    step_idx::Int
end

function CommonSolve.step!(forward::SimulatorForwardIntegrator)
    if forward.step_idx > length(forward.tstops)
        return nothing
    end
    # extract fields from forward integrator and compute dt
    prob = forward.prob
    integrator = forward.integrator
    t = forward.tstops[forward.step_idx]
    dt = prob.obs_to_prob_time(t) - integrator.t
    if dt > 0
        # step to next t if dt > 0
        step!(integrator, dt, true)
    end
    # iterate over observables and update those for which t is a sample point
    for obs in prob.observables
        if t ∈ samplepoints(obs)
            observe!(obs, integrator)
        end
    end
    # increment step index
    forward.step_idx += 1
    return nothing
end

function CommonSolve.init(prob::SimulatorForwardProblem, ode_alg; kwargs...)
    # collect and combine sample points from all obsevables
    t_points = sort(unique(union(map(samplepoints, prob.observables)...)))
    # reinitialize forward problem with new parameters
    newprob = remake(prob, p=p)
    # initialize integrator with built-in saving disabled
    integrator = init(newprob.prob, ode_alg; kwargs...)
    # initialize observables
    for obs in prob.observables
        init!(obs, integrator)
    end
    return SimulatorForwardIntegrator(prob, integrator, t_points, 1)
end

"""
    solve(prob::SimulatorForwardProblem, ode_alg, p; kwargs...)

Solves the forward problem using the given diffeq algorithm and parameters `p`.
"""
function CommonSolve.solve(prob::SimulatorForwardProblem, ode_alg; p=prob.prob.p, kwargs...)
    forward = CommonSolve.init(prob::SimulatorForwardProblem, ode_alg; p, kwargs...)
    # iterate until end
    for i in forward end
    return SimulatorForwardSolution(forward.prob, forward.integrator.sol)
end