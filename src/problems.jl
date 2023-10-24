#### Forward problems ####

"""
    SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem

Represents a "forward" problem from parameters/initial conditions to output `SimulatorObservable`s.
"""
struct SimulatorForwardProblem{probType,obsType,configType,names} <: SciMLBase.AbstractSciMLProblem
    prob::probType
    observables::NamedTuple{names,obsType}
    config::configType
end

function SciMLBase.remaker_of(forward_prob::SimulatorForwardProblem)
    function remake_forward_prob(;
        prob=forward_prob,
        observables=forward_prob.observables,
        config=forward_prob.config,
        kwargs...
    )
        return SimulatorForwardProblem(remake(prob; kwargs...), observables, config)
    end
end

# DiffEqBase dispatches to make solve/init interface work correctly
DiffEqBase.check_prob_alg_pairing(prob::SimulatorForwardProblem, alg) = DiffEqBase.check_prob_alg_pairing(prob.prob, alg)
DiffEqBase.isinplace(prob::SimulatorForwardProblem) = DiffEqBase.isinplace(prob.prob)

# Overload property methods to forward properties from `prob` field
Base.propertynames(prob::SimulatorForwardProblem) = (:prob, :observables, :config, propertynames(prob.prob)...)
function Base.getproperty(prob::SimulatorForwardProblem, sym::Symbol)
    if sym ∈ (:prob,:observables,:config)
        return getfield(prob, sym)
    end
    return getproperty(getfield(prob, :prob), sym)
end

"""
    SimulatorForwardSolution{TSol}

Solution for a `SimulatorForwardProblem` that wraps the underlying `DESolution`.
"""
struct SimulatorForwardSolution{TSol}
    prob::SimulatorForwardProblem
    sol::TSol
end

#### Inverse/inference problems ####

"""
    SimulatorInferenceProblem{priorType<:JointPrior,uType,algType,likType,dataType,names} <: SciMLBase.AbstractOptimizationProblem{false}

Represents a generic simulation-based Bayesian inference problem for finding the posterior distribution over model parameters given some observed data.
"""
struct SimulatorInferenceProblem{priorType<:JointPrior,uType,algType,likType,dataType,names} <: SciMLBase.AbstractOptimizationProblem{false}
    u0::uType
    forward_prob::SimulatorForwardProblem
    forward_solver::algType
    prior::priorType
    param_map::ParameterMapping
    likelihoods::NamedTuple{names,likType}
    data::NamedTuple{names,dataType}
end
(::Type{SimulatorInferenceProblem})(
    prob::SimulatorInferenceProblem;
    u0=prob.u0,
    forward_prob=prob.forward_prob,
    forward_solver=prob.forward_solver,
    prior=prob.prior,
    param_map=prob.param_map,
    likelihoods=prob.likelihoods,
    data=prob.data,
) = SimulatorInferenceProblem(u0, forward_prob, forward_solver, prior, param_map, likelihoods, data)

"""
    SimulatorInferenceProblem(
        prob::SimulatorForwardProblem,
        prior::AbstractPrior,
        lik_with_data::Pair{<:Likelihood,<:AbstractArray}...;
        param_map=ParameterMapping(prior),
    )

Constructs a `SimulatorInferenceProblem` from the given forward problem, prior, and likelihood/data pairs.
"""
function SimulatorInferenceProblem(
    forward_prob::SimulatorForwardProblem,
    forward_solver::DEAlgorithm,
    prior::AbstractPrior,
    lik_with_data::Pair{<:Likelihood,<:AbstractArray}...;
    param_map=ParameterMapping(prior),
)
    likelihoods = map(first, lik_with_data)
    data = map(last, lik_with_data)
    # convert to named tuple with names as keys
    likelihoods_with_names = (; map(l -> nameof(l) => l, likelihoods)...)
    data_with_names = (; map(Pair, keys(likelihoods_with_names), data)...)
    joint_prior = JointPrior(prior, likelihoods...)
    u0 = zero(rand(joint_prior))
    SimulatorInferenceProblem(u0, forward_prob, forward_solver, joint_prior, param_map, likelihoods_with_names, data_with_names)
end

Base.names(::SimulatorInferenceProblem{TP,TL,TD,names}) where {TP,TL,TD,names} = names

function Base.getproperty(prob::SimulatorInferenceProblem, sym::Symbol)
    if sym ∈ (:u0,:forward_prob,:forward_solver,:prior,:param_map,:likelihoods,:data)
        return getfield(prob, sym)
    end
    return getproperty(getfield(prob, :forward_prob), sym)
end

function logprob(inference_prob::SimulatorInferenceProblem, _x::AbstractVector; transform=false, solve_kwargs...)
    x = ComponentVector(getdata(_x), getaxes(inference_prob.u0))
    logp = zero(eltype(x))
    # transform from unconstrained space if necessary
    if transform
        logp += logprob(inference_prob.prior, x)
        logp += logprob(inference_prob.param_map, x)
        z = inference_prob.param_map(x)
    else
        z = x
    end
    # solve forward problem
    _ = solve(inference_prob.forward_prob, inference_prob.forward_solver; p=z.model, solve_kwargs...)
    # compute the likelihood distributions from the observables and likelihood parameters
    lik_dists = map(l -> l(getproperty(z, nameof(l))), inference_prob.likelihoods)
    # compute and sum the log densities
    return sum(map((x,D) -> logpdf(D,x), inference_prob.data, lik_dists))
end

# solve interface method stub
function CommonSolve.solve(
    inference_prob::SimulatorInferenceProblem,
    alg::SimulatorInferenceAlgorithm,
    args...;
    kwargs...
)
    error("solve not implemented for algorithm $(typeof(alg)) on $(typeof(inference_prob))")
end

# log density interface

LogDensityProblems.capabilities(::Type{<:SimulatorInferenceProblem}) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.dimension(inference_prob::SimulatorInferenceProblem) = length(inference_prob.u0)

LogDensityProblems.logdensity(inference_prob::SimulatorInferenceProblem, x) = logprob(inference_prob, x)
