"""
    EnsembleState

Base type for ensemble inference algorithm state implementations.
"""
abstract type EnsembleState end

"""
    EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,kwargTypes}

Generic implementation of an iterative solver for any ensemble-based algorithm. Uses
the SciML `EnsembleProblem` interface to automatically parallelize forward runs over
the ensemble.
"""
mutable struct EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,argTypes,kwargTypes}
    sol::SimulatorInferenceSolution{algType,probType} # solution
    alg::algType # inference algorithm
    ensalg::ensalgType # ensemble execution algorithm
    state::stateType # algorithm state
    loglik::Vector # log likelihoods for each iteration
    logprior::Vector # log prior probabilities for each iteration
    prob_func::Function # problem generator
    output_func::Function # output function
    itercallback::Function # iteration callback
    verbose::Bool # true if verbose output should be printed to sdtout
    retcode::ReturnCode.T # return code status
    solve_args::argTypes # positional args passed to the forward solve
    solve_kwargs::kwargTypes # keyword args passed to the forward solve
end

################################
# Ensemble algorithm interface #
################################

"""
    get_ensemble(state::EnsembleState)

Retrieves the current ensemble matrix from the given `EnsembleState`. Must be implemented for
each ensemble algorithm state type.
"""
get_ensemble(state::EnsembleState) = error("get_ensemble(state) not implemented for $(typeof(state))")

"""
    isiterative(alg::EnsembleInferenceAlgorithm)

Returns `true` if the given ensemble inference algorithm is iterative, `false` otherwise.
Default implementation returns `false` (non-iterative).
"""
isiterative(alg::EnsembleInferenceAlgorithm) = false

"""
    hasconverged(alg::EnsembleInferenceAlgorithm, state::EnsembleState)

Should return `true` when `alg` at the current `state` has converged, `false` otherwise. Must be implemented
for each ensemble algorithm state type.
"""
hasconverged(alg::EnsembleInferenceAlgorithm, state::EnsembleState) = error("hasconverged not implemented for alg $(typeof(alg))")

"""
    initialstate(
        alg::EnsembleInferenceAlgorithm,
        ens::AbstractMatrix,
        obs::AbstractVector,
        obscov::AbstractMatrix;
        rng::AbstractRNG=Random.GLOBAL_RNG
    )

Constructs the initial ensemble state for the given algorithm and observations.
"""
initialstate(
    alg::EnsembleInferenceAlgorithm,
    prior::AbstractSimulatorPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obscov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG
) = error("intialstate not implemented for $(typeof(alg))")

"""
    ensemblestep!(solver::EnsembleSolver{algType}) where {algType}

Executes a single ensemble step (forward solve + update) for the given algorithm type. Must be implemented
by all ensemble algorithm implementations.
"""
ensemblestep!(::EnsembleSolver{algType}) where {algType} = error("not implemented for alg of type $algType")

"""
    finalize!(solver::EnsembleSolver)

Finalizes the solver state after iteration has completed. Default implementation runs `ensemble_solve`
on the current ensemble state and stores the results in `sol.storage`.
"""
function finalize!(solver::EnsembleSolver)
    out = ensemble_forward(solver)
    return if isiterative(solver.alg)
        store!(solver.sol.storage, get_ensemble(solver.state), out.observables, iter=solver.state.iter + 1)
    else
        store!(solver.sol.storage, get_ensemble(solver.state), out.observables)
    end
end

################################

init(inference_prob::SimulatorInferenceProblem, alg::EnsembleInferenceAlgorithm, args...; kwargs...) = init(inference_prob, alg, nothing, args...; kwargs...)

"""
    init(::SimulatorInferenceProblem, ::EKS, ensemble_alg; kwargs...)

Initializes EKS for the given `SimulatorInferenceProblem` using the Ensemble Kalman Sampling algorithm. This method automatically
constructs an `EnsembleKalmanProcess` from the given inference problem and `named_data` pairs. If `initial_ens` is not provided,
the initial ensemble is sampled from the prior.
"""
function init(
    inference_prob::SimulatorInferenceProblem,
    alg::EnsembleInferenceAlgorithm,
    ensalg::Union{Nothing,EnsembleAlgorithm}=EnsembleThreads(),
    solve_args...;
    prob_func=(prob, p) -> remake(prob; p),
    validator_func=(sol, i) -> OK,
    output_func=ensemble_output_func(inference_prob.forward_prob),
    obs_cov_func=obscov,
    initial_ens=nothing,
    ensemble_size::Integer=isnothing(initial_ens) ? 128 : size(initial_ens, 2),
    itercallback=state -> true,
    storage=SimulationArrayStorage(),
    verbose=true,
    rng=Random.default_rng(),
    solve_kwargs...
)
    # extract model prior (i.e. ignoring likelihood parameters)
    model_prior = inference_prob.prior.model
    # construct transform from model prior
    constrained_to_unconstrained = bijector(inference_prob.prior.model)
    # sample initial ensemble
    if isnothing(initial_ens)
        samples = sample(rng, model_prior, ensemble_size)
        # apply transform to samples and then concatenate on second axis
        initial_ens = reduce(hcat, map(constrained_to_unconstrained, samples))
    else
        initial_ens = reduce(hcat, map(constrained_to_unconstrained, eachcol(initial_ens)))
    end
    # extract observations from likelihood terms
    likelihoods = values(inference_prob.likelihoods)
    obs_mean = reduce(vcat, map(l -> vec(l.data), likelihoods))
    obs_cov = obs_cov_func(likelihoods...)
    ensemble_size = size(initial_ens, 2)
    # construct initial state
    state = initialstate(alg, model_prior, initial_ens, obs_mean, obs_cov; rng)
    inference_sol = SimulatorInferenceSolution(inference_prob, alg, storage, state)
    return EnsembleSolver(
        inference_sol,
        alg,
        ensalg,
        state,
        [],
        [],
        prob_func,
        output_func,
        itercallback,
        verbose,
        ReturnCode.Default,
        solve_args,
        (; prob_func, validator_func, solve_kwargs...),
    )
end

function step!(solver::EnsembleSolver)
    # if retcode is already set to a terminal value, return immediately
    if solver.retcode != ReturnCode.Default
        return false
    end
    alg = solver.alg
    sol = solver.sol
    state = solver.state
    state.iter += 1
    # ensemble step
    out = ensemblestep!(solver)
    # set result
    sol.result = state
    # store observables
    store!(sol.storage, get_ensemble(state), out.observables, iter=state.iter)
    # iteration callback
    callback_retval = solver.itercallback(state)
    # check convergence
    converged = !isiterative(alg) || hasconverged(alg, state)
    maxiters_reached = isiterative(alg) ? state.iter >= alg.maxiters : false
    retcode = if converged
        ReturnCode.Success
    elseif maxiters_reached
        ReturnCode.MaxIters
    elseif !isnothing(callback_retval) && !callback_retval
        ReturnCode.Terminated
    else
        ReturnCode.Default
    end
    solver.retcode = retcode
    return retcode == ReturnCode.Default
end

function solve!(solver::EnsembleSolver)
    # step until convergence
    while step!(solver)
    end
    # finalize
    finalize!(solver)
    # return inference solution
    return solver.sol
end

function ensemble_forward(solver::EnsembleSolver)
    inference_prob = solver.sol.prob
    # get current parameter ensemble
    θ = get_ensemble(solver.state)
    # map unconstrained parameters to constrained space
    param_map = unconstrained_forward_map(inference_prob.prior.model)
    p = reduce(hcat, map(param_map, eachcol(θ)))
    # rebuild forward problem with the constrained parameter ensemble
    forward_prob = remake(inference_prob.forward_prob, p=p)
    # solve the ensemble forward problem
    enssol = solve(forward_prob, inference_prob.forward_solver, solver.ensalg, solver.solve_args...; solver.solve_kwargs...)
    return ensemble_outputs(inference_prob, enssol)
end

# Non-batched simulator, ensemble solve
function ensemble_outputs(inference_prob::SimulatorInferenceProblem, sol::EnsembleSolution)
    # extract prediction vector for combined likelihoods
    pred = mapreduce(hcat, sol.u) do result
        # retrieve observable for each likelihood and flatten into a vector
        observables = map(name -> result.observables[name], keys(inference_prob.likelihoods))
        reduce(vcat, map(obs -> vec(obs), observables))
    end
    # extract observable values
    observables = ntreduce(enscat, map(result -> result.observables, sol.u))
    return (; pred, observables)
end

# Batched simulator
function ensemble_outputs(inference_prob::SimulatorInferenceProblem, sol::SimulatorForwardSolution)
    # extract prediction vector for combined likelihoods
    pred = mapreduce(vcat, keys(inference_prob.likelihoods)) do name
        arr = getvalue(sol.prob.observables[name])
        # flatten all but the last (batch) axis
        reshape(arr, :, length(last(axes(arr))))
    end
    # extract observable values
    observables = map(getvalue, result.observables)
    return (; pred, observables)
end
