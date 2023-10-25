const DEProblem = SciMLBase.DEProblem
const DEIntegrator = SciMLBase.DEIntegrator

struct SimulatorDEConfig{F}
    obs_to_prob_time::F
end

default_time_converter(prob::DEProblem) = identity

function SimulatorForwardProblem(prob::DEProblem, observables::SimulatorObservable...; obs_to_prob_time=default_time_converter(prob))
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, SimulatorDEConfig(obs_to_prob_time))
end

"""
    SimulatorForwardDEIntegrator{algType,uType,tType,iip,integratorType<:DEIntegrator{algType,iip,uType,tType}} <: DEIntegrator{algType,iip,uType,tType}

Specialized integrator type that wraps a `SciMLBase.DEIntegrator` and controls the stepping procedure such that each observable sample point is hit.
"""
mutable struct SimulatorForwardDEIntegrator{algType,uType,tType,iip,integratorType<:DEIntegrator{algType,iip,uType,tType}} <: DEIntegrator{algType,true,uType,tType}
    prob::SimulatorForwardProblem
    integrator::integratorType
    tstops::Vector{tType}
    step_idx::Int
end

Base.propertynames(integrator::SimulatorForwardDEIntegrator) = (:prob,:integrator,:tstops,:step_idx,propertynames(integrator.integrator)...)
function Base.getproperty(integrator::SimulatorForwardDEIntegrator, sym::Symbol)
    if sym ∈ (:prob,:integrator,:tstops,:step_idx)
        return getfield(integrator, sym)
    else
        return getproperty(getfield(integrator,:integrator), sym)
    end
end
function Base.setproperty!(integrator::SimulatorForwardDEIntegrator, sym::Symbol, value)
    if sym ∈ (:prob,:integrator,:tstops,:step_idx)
        return setfield!(integrator, sym, value)
    else
        return setproperty!(getfield(integrator, :integrator), sym, value)
    end
end

function CommonSolve.step!(forward::SimulatorForwardDEIntegrator)
    if forward.step_idx > length(forward.tstops)
        return nothing
    end
    # extract fields from forward integrator and compute dt
    prob = forward.prob
    integrator = forward.integrator
    t = forward.tstops[forward.step_idx]
    dt = prob.config.obs_to_prob_time(t) - integrator.t
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

function CommonSolve.init(prob::SimulatorForwardProblem{<:DEProblem}, ode_alg; p=prob.prob.p, kwargs...)
    # collect and combine sample points from all obsevables
    t_points = sort(unique(union(map(samplepoints, prob.observables)...)))
    # reinitialize forward problem with new parameters
    newprob = remake(prob, p=p)
    # initialize integrator with built-in saving disabled
    integrator = init(newprob.prob, ode_alg; kwargs...)
    # initialize observables
    for obs in prob.observables
        initialize!(obs, integrator)
    end
    return SimulatorForwardDEIntegrator(prob, integrator, t_points, 1)
end

"""
    solve(prob::SimulatorForwardProblem, ode_alg, p; kwargs...)

Solves the forward problem using the given diffeq algorithm and parameters `p`.
"""
function CommonSolve.solve(prob::SimulatorForwardProblem{<:DEProblem}, ode_alg; p=prob.prob.p, kwargs...)
    forward = CommonSolve.init(prob::SimulatorForwardProblem, ode_alg; p, kwargs...)
    # iterate until end
    for i in forward end
    return SimulatorForwardSolution(forward.prob, forward.integrator.sol)
end