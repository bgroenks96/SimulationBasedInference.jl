#### Inverse/inference problems ####

"""
    SimulatorInferenceProblem{priorType<:JointPrior,uType,algType,likType} <: SciMLBase.AbstractSciMLProblem

Represents a generic simulation-based Bayesian inference problem for finding the posterior distribution over model parameters given some observed data.
"""
struct SimulatorInferenceProblem{priorType<:JointPrior,uType,algType,likType} <: SciMLBase.AbstractSciMLProblem
    u0::uType
    forward_prob::SimulatorForwardProblem
    forward_solver::algType
    prior::priorType
    param_map::ParameterMapping
    likelihoods::likType
    metadata::Dict
end

"""
    SimulatorInferenceProblem(
        prob::SimulatorForwardProblem,
        prior::AbstractPrior,
        likelihoods::SimulatorLikelihood...;
        param_map=ParameterMapping(prior),
        metadata::Dict=Dict(),
    )

Constructs a `SimulatorInferenceProblem` from the given forward problem, prior, and likelihoods.
The parameter mapping is by default auto-determined from the prior. Additional user-specified
metadata may be included in the `metadata` dictionary.
"""
function SimulatorInferenceProblem(
    forward_prob::SimulatorForwardProblem,
    forward_solver::DEAlgorithm,
    prior::AbstractPrior,
    likelihoods::SimulatorLikelihood...;
    param_map=ParameterMapping(prior),
    metadata::Dict=Dict(),
)
    joint_prior = JointPrior(prior, likelihoods...)
    u0 = zero(rand(joint_prior))
    return SimulatorInferenceProblem(u0, forward_prob, forward_solver, joint_prior, param_map, with_names(likelihoods), metadata)
end

SciMLBase.isinplace(prob::SimulatorInferenceProblem) = false

function SciMLBase.remaker_of(prob::SimulatorInferenceProblem)
    function remake(;
        u0=prob.u0,
        forward_prob=prob.forward_prob,
        forward_solver=prob.forward_solver,
        prior=prob.prior,
        param_map=prob.param_map,
        likelihoods=prob.likelihoods,
        metadata=prob.metadata,
    )
        SimulatorInferenceProblem(u0, forward_prob, forward_solver, prior, param_map, likelihoods, metadata)
    end
end

Base.names(::SimulatorInferenceProblem{TP,TL,TD,names}) where {TP,TL,TD,names} = names

Base.propertynames(prob::SimulatorInferenceProblem) = (fieldnames(typeof(prob))..., propertynames(getfield(prob, :forward_prob))...)
function Base.getproperty(prob::SimulatorInferenceProblem, sym::Symbol)
    if sym ∈ fieldnames(typeof(prob))
        return getfield(prob, sym)
    end
    return getproperty(getfield(prob, :forward_prob), sym)
end

"""
    logjoint(inference_prob::SimulatorInferenceProblem, u::AbstractVector; transform=false, forward_solve=true, solve_kwargs...)

Evaluate the the log-joint density components `log(p(D|x))` and `log(p(u))` where `D` is the data and `u` are the parameters.
If `transform=true`, `x` is transformed by the `param_map` defined on the given inference problem before evaluating
the density.
"""
function logjoint(
    inference_prob::SimulatorInferenceProblem,
    u::AbstractVector;
    transform=false,
    forward_solve=true,
    solve_kwargs...
)
    uvec = ComponentVector(getdata(u), getaxes(inference_prob.u0))
    logprior = zero(eltype(u))
    # transform from unconstrained space if necessary
    if transform
        ϕ = inference_prob.param_map(uvec)
        # add density change due to transform
        logprior += logdensity(inference_prob.param_map, uvec)
    else
        ϕ = uvec
    end
    logprior += logdensity(inference_prob.prior, ϕ)
    # solve forward problem;
    if forward_solve
        # we can discard the result of solve since the observables are already stored in the likelihoods.
        _ = solve(inference_prob.forward_prob, inference_prob.forward_solver; p=ϕ.model, solve_kwargs...)
    else
        # check that parameters match
        @assert all(ϕ.model .≈ inference_prob.forward_prob.p) "forward problem model parameters do not match the given parameter"
    end
    # compute the likelihood distributions from the observables and likelihood parameters
    loglik = sum(map(l -> loglikelihood(l, getproperty(ϕ, nameof(l))), inference_prob.likelihoods))
    return (; loglik, logprior)
end

function logjoint(
    inference_prob::SimulatorInferenceProblem,
    forward_sol::SimulatorForwardSolution,
    u::AbstractVector;
    transform=false,
    solve_kwargs...
)
    observables = forward_sol.prob.observables
    new_likelihoods = map(l -> remake(l, obs=getproperty(observables, nameof(l))), inference_prob.likelihoods)
    new_inference_prob = remake(inference_prob; forward_prob=forward_sol.prob, likelihoods=new_likelihoods)
    return logjoint(new_inference_prob, u; transform, forward_solve=false, solve_kwargs...)
end

"""
    logdensity(inference_prob::SimulatorInferenceProblem, x; kwargs...)

Calls `logjoint` with the given arguments and sums the resulting likelihood and prior log-densities.
"""
logdensity(inference_prob::SimulatorInferenceProblem, x; kwargs...) = sum(logjoint(inference_prob, x; kwargs...))

# log density interface

LogDensityProblems.capabilities(::Type{<:SimulatorInferenceProblem}) = LogDensityProblems.LogDensityOrder{0}()

LogDensityProblems.dimension(inference_prob::SimulatorInferenceProblem) = length(inference_prob.u0)

"""
    SimulatorInferenceSolution{probType,resultType}

Generic container for solutions to `SimulatorInferenceProblem`s. The type of `inference_result` is method dependent
and should generally correspond to the final state or product of the inference algorithm (e.g. posterior sampels).
The vectors `inputs` and `outputs` should be populated with input parameters and their corresponding output solutions
respectively.
"""
mutable struct SimulatorInferenceSolution{probType,resultType}
    prob::probType
    inputs::Vector
    outputs::Vector
    inference_result::resultType
end
