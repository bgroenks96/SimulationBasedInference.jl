abstract type EnsembleState end

"""
    EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,kwargTypes}

Generic implementation of an iterative solver for any ensemble-based algorithm. Uses
the SciML `EnsembleProblem` interface to automatically parallelize forward runs over
the ensemble.
"""
mutable struct EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,kwargTypes}
    sol::SimulatorInferenceSolution{algType,probType}
    alg::algType
    ensalg::ensalgType
    state::stateType
    prob_func::Function # problem generator
    output_func::Function # output function
    pred_func::Function # prediction function
    itercallback::Function # iteration callback
    verbose::Bool
    retcode::ReturnCode.T
    solve_kwargs::kwargTypes
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
    ensemblestep!(solver::EnsembleSolver{algType})

Executes a single ensemble step (forward solve + update) for the given algorithm type. Must be implemented
by all ensemble algorithm implementations.
"""
ensemblestep!(::EnsembleSolver{algType}) where {algType} = error("not implemented for alg of type $algType")

"""
    finalize!(solver::EnsembleSolver)

Finalizes the solver state after iteration has completed. Default implementation runs `ensemble_solve`
on the current ensemble state and pushes the results to `sol.outputs`.
"""
function finalize!(solver::EnsembleSolver)
    # model parameter forward map
    param_map = unconstrained_forward_map(solver.sol.prob.prior.model)
    out = ensemble_solve(
        solver.state,
        solver.sol.prob.forward_prob,
        solver.ensalg,
        solver.sol.prob.forward_solver,
        param_map;
        prob_func=solver.prob_func,
        output_func=solver.output_func,
        pred_func=solver.pred_func,
        solver.solve_kwargs...
    )

    if isiterative(solver.alg)
        store!(solver.sol.storage, get_ensemble(solver.state), out.observables, iter=solver.state.iter+1)
    else
        store!(solver.sol.storage, get_ensemble(solver.state), out.observables)
    end
end

################################

"""
    init(::SimulatorInferenceProblem, ::EKS, ensemble_alg; kwargs...)

Initializes EKS for the given `SimulatorInferenceProblem` using the Ensemble Kalman Sampling algorithm. This method automatically
constructs an `EnsembleKalmanProcess` from the given inference problem and `named_data` pairs. If `initial_ens` is not provided,
the initial ensemble is sampled from the prior.
"""
function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::EnsembleInferenceAlgorithm,
    ensalg::SciMLBase.EnsembleAlgorithm=EnsembleThreads();
    prob_func=(prob,p) -> remake(prob, p=p),
    output_func=(sol,i,iter) -> (sol, false),
    pred_func=default_pred_func(inference_prob),
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
    inference_sol = SimulatorInferenceSolution(inference_prob, alg, storage, nothing)
    return EnsembleSolver(
        inference_sol,
        alg,
        ensalg,
        state,
        prob_func,
        output_func,
        pred_func,
        itercallback,
        verbose,
        ReturnCode.Default,
        solve_kwargs,
    )
end

function CommonSolve.step!(solver::EnsembleSolver)
    # if retcode is already set to a terminal value, return immediately
    if solver.retcode != ReturnCode.Default
        return false
    end
    alg = solver.alg
    sol = solver.sol
    state = solver.state
    state.iter += 1
    # ensemble step
    ensemblestep!(solver)
    # set result
    sol.result = state
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

function CommonSolve.solve!(solver::EnsembleSolver)
    # step until convergence
    while CommonSolve.step!(solver) end
    # finalize
    finalize!(solver)
    # return inference solution
    return solver.sol
end

"""
    ensemble_solve(
        state::EnsembleState,
        initial_prob::SciMLBase.AbstractSciMLProblem,
        ensalg::SciMLBase.BasicEnsembleAlgorithm,
        dealg::Union{Nothing,SciMLBase.AbstractSciMLAlgorithm},
        param_map,
        prob_func,
        output_func,
        pred_func;
        solve_kwargs...
    )

Performs a single step/iteration for the given ensemble and returns a named tuple
`(; pred, sol)` where `sol` are the full ensemble forward solutions and `pred` is
the prediction matrix produced by `pred_func`.
"""
function ensemble_solve(
    ens::AbstractMatrix,
    initial_prob::SciMLBase.AbstractSciMLProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    dealg::Union{Nothing,SciMLBase.AbstractSciMLAlgorithm},
    param_map;
    iter::Integer=1,
    prob_func=(prob,p) -> remake(prob, p=p),
    output_func=(sol,i,iter) -> (sol, false),
    pred_func=default_pred_func(initial_prob),
    solve_kwargs...
)
    Θ = ens
    N_ens = size(Θ,2)
    ensprob = EnsembleProblem(
        initial_prob;
        prob_func=(prob,i,repeat) -> prob_func(prob, param_map(ens[:,i])),
        output_func=(sol,i) -> output_func(sol, i, iter)
    )

    enssol = solve(ensprob, dealg, ensalg; trajectories=N_ens, solve_kwargs...)
    pred = reduce(hcat, map((i,out) -> pred_func(out, i, iter), 1:N_ens, enssol.u))
    observables = map(enssol) do sol
        # extract observables data
        map(getvalue, sol.prob.observables)
    end
    return (; pred, observables)
end
ensemble_solve(state::EnsembleState, args...; kwargs...) = ensemble_solve(get_ensemble(state), args...; iter=state.iter, kwargs...)

function default_pred_func(prob::SimulatorInferenceProblem; verify_return_code=false)
    function predict_likelihoods(sol::SimulatorForwardSolution, i, iter)
        if isa(sol.sol, SciMLBase.AbstractSciMLSolution) && verify_return_code
            # check forward solver return code
            @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
        end
        # retrieve observable for each likelihood and flatten the results into a vector
        observables = map(name -> sol.prob.observables[name], keys(prob.likelihoods))
        return reduce(vcat, map(obs -> vec(getvalue(obs)), observables))
    end
end

function default_pred_func(::SimulatorForwardProblem; verify_return_code=false)
    function predict_likelihoods(sol::SimulatorForwardSolution, i, iter)
        if isa(sol.sol, SciMLBase.AbstractSciMLSolution) && verify_return_code
            # check forward solver return code
            @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
        end
        # retrieve observable for each likelihood and flatten the results into a vector
        return reduce(vcat, map(obs -> vec(getvalue(obs)), sol.prob.observables))
    end
end
