const AbstractODEProblem = SciMLBase.AbstractODEProblem
const AbstractODEIntegrator = SciMLBase.AbstractODEIntegrator

struct SimulatorODEConfig{F,T}
    stepfunc::F
    obs_to_prob_time::T
end

default_time_converter(::AbstractODEProblem) = identity

function SimulatorForwardProblem(
    prob::AbstractODEProblem,
    observables::SimulatorObservable...;
    stepfunc=step!,
    obs_to_prob_time=default_time_converter(prob)
)
    named_observables = (; map(x -> nameof(x) => x, observables)...)
    return SimulatorForwardProblem(prob, named_observables, SimulatorODEConfig(stepfunc, obs_to_prob_time))
end

function SimulatorForwardProblem(
    prob::AbstractODEProblem,
    p::AbstractVector,
    observables::SimulatorObservable...;
    stepfunc=step!,
    obs_to_prob_time=default_time_converter(prob)
)
    return SimulatorForwardProblem(remake(prob, p=p), observables...; stepfunc, obs_to_prob_time)
end


function ODEObservable(name::Symbol, prob::AbstractODEProblem, tsave; obsfunc=identity, kwargs...)
    return SimulatorObservable(name, integrator -> obsfunc(integrator.u), prob.tspan[1], tsave, size(prob.u0); kwargs...)
end

"""
    SimulatorODEForwardSolver{algType,uType,tType,iip,integratorType<:AbstractODEIntegrator{algType,iip,uType,tType}} <: AbstractODEIntegrator{algType,iip,uType,tType}

Specialized integrator type that wraps a SciML ODE integrator and controls the stepping procedure such that each observable sample point is hit.
"""
mutable struct SimulatorODEForwardSolver{algType,uType,tType,iip,integratorType<:AbstractODEIntegrator{algType,iip,uType,tType}} <: AbstractODEIntegrator{algType,true,uType,tType}
    prob::SimulatorForwardProblem
    integrator::integratorType
    tstops::Vector
    step_idx::Int
end

SciMLBase.done(solver::SimulatorODEForwardSolver) = SciMLBase.done(solver.integrator)
SciMLBase.postamble!(solver::SimulatorODEForwardSolver) = SciMLBase.postamble!(solver.integrator)

# forwarding property dispatches to nested integrator
Base.propertynames(integrator::SimulatorODEForwardSolver) = (:prob,:integrator,:tstops,:step_idx,propertynames(integrator.integrator)...)
function Base.getproperty(integrator::SimulatorODEForwardSolver, sym::Symbol)
    if sym ∈ fieldnames(typeof(integrator))
        return getfield(integrator, sym)
    else
        return getproperty(getfield(integrator,:integrator), sym)
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
    init(forward_prob::SimulatorForwardProblem{<:AbstractODEProblem}, ode_alg; p=forward_prob.prob.p, saveat=[], solve_kwargs...)

Initializes a `SimulatorODEForwardSolver` for the given forward problem and ODE integrator algorithm. Additional keyword arguments are
passed through to the integrator `init` implementation.
"""
function CommonSolve.init(
    forward_prob::SimulatorForwardProblem{<:AbstractODEProblem},
    ode_alg;
    p=forward_prob.prob.p,
    saveat=[],
    save_everystep=false,
    copy_observables=false,
    solve_kwargs...
)
    # collect and combine sample points from all obsevables
    t_sample = map(sampletimes, forward_prob.observables)
    t_sample_all = forward_prob.config.obs_to_prob_time.(sort(unique(union(t_sample...)))) 
    t_points = if isempty(t_sample_all) || t_sample_all[end] < forward_prob.tspan[end]
        adstrip.(vcat(t_sample_all, [forward_prob.tspan[end]]))
    else
        adstrip.(t_sample_all)
    end
    # reinitialize inner problem with new parameters
    newprob = remake(forward_prob; p, copy_observables)
    # initialize integrator with built-in saving disabled
    integrator = init(newprob.prob, ode_alg; saveat, save_everystep, solve_kwargs...)
    # initialize observables
    for obs in newprob.observables
        initialize!(obs, integrator)
    end
    # iterate over observables and update those for which t0 is a sample point
    for obs in newprob.observables
        if integrator.t ∈ map(newprob.config.obs_to_prob_time, sampletimes(obs))
            observe!(obs, integrator)
        end
    end
    return SimulatorODEForwardSolver(newprob, integrator, t_points, 1)
end

function CommonSolve.step!(forward::SimulatorODEForwardSolver)
    # extract fields from forward integrator and compute dt
    prob = forward.prob
    integrator = forward.integrator
    forwardstep! = prob.config.stepfunc
    t = forward.tstops[forward.step_idx]
    dt = adstrip(t - integrator.t)
    # if there are no more stopping points, just forward to the integrator and return
    if forward.step_idx > length(forward.tstops)
        return forwardstep!(forward.integrator)
    end
    # otherwise, evaluate the next step and observables if dt > 0
    retval = if dt > zero(dt)
        forwardstep!(integrator, dt, true)
    else
        nothing
    end
    # iterate over observables and update those for which t is a sample point
    for obs in prob.observables
        if t ∈ map(prob.config.obs_to_prob_time, sampletimes(obs))
            observe!(obs, integrator)
        end
    end
    # increment step index
    forward.step_idx += 1
    return retval
end

"""
    solve!(prob::SimulatorForwardProblem, ode_alg, p; kwargs...)

Solves the forward problem using the given diffeq algorithm and parameters `p`.
"""
function CommonSolve.solve!(forwardsolver::SimulatorODEForwardSolver)
    # iterate until end
    for i in forwardsolver end
    return SimulatorForwardSolution(forwardsolver.prob, forwardsolver.integrator.sol)
end
