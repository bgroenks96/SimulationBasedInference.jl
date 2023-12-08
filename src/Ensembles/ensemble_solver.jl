abstract type EnsembleState end

"""
    EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,kwargTypes}

Generic implementation of an iterative solver for any ensemble-based algorithm. Uses
the SciML `EnsembleProblem` interface to automatically parallelize forward runs over
the ensemble.
"""
mutable struct EnsembleSolver{algType,probType,ensalgType,stateType<:EnsembleState,kwargTypes}
    sol::SimulatorInferenceSolution{probType}
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
get_ensemble(state::EnsembleState) = error("get_ensemble not implemented for state type $(typeof(state))")

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
    ens::AbstractMatrix,
    obs::AbstractVector,
    obscov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG
) = error("intialstate not implemented for $(typeof(alg))")

"""
    ensemble_step!(solver::EnsembleSolver{algType})

Executes a single ensemble step (forward solve + update) for the given algorithm type. Must be implemented
by all ensemble algorithm implementations.
"""
ensemble_step!(solver::EnsembleSolver{algType}) where {algType} = error("not implemented for alg of type $algType")

################################

"""
    init(::SimulatorInferenceProblem, ::EKS, ensemble_alg; kwargs...)

Initializes EKS for the given `SimulatorInferenceProblem` using the Ensemble Kalman Sampling algorithm. This method automatically
constructs an `EnsembleKalmanProcess` from the given inference problem and `named_data` pairs. If `initial_ens` is not provided,
the initial ensemble is sampled from the prior.
"""
function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::EnsembleInferenceAlgorithm;
    prob_func=(prob,p) -> remake(prob, p=p),
    output_func=(sol,i,iter) -> (sol, false),
    pred_func=default_forward_solve_pred,
    rng=Random.default_rng(),
    initial_ens=nothing,
    itercallback=state -> true,
    verbose=true,
    solve_kwargs...
)
    # sample initial ensemble
    if isnothing(initial_ens)
        # extract model prior (i.e. ignoring likelihood parameters)
        model_prior = inference_prob.prior.model
        # construct transform from model prior
        constrained_to_unconstrained = bijector(inference_prob.prior.model)
        samples = sample(rng, model_prior, alg.n_ens)
        # apply transform to samples and then concatenate on second axis
        initial_ens = reduce(hcat, map(constrained_to_unconstrained, samples))
    end
    # extract observations from likelihood terms
    likelihoods = values(inference_prob.likelihoods)
    obs_mean = reduce(vcat, map(l -> vec(l.data), likelihoods))
    obs_cov = alg.obs_cov(likelihoods...)
    n_ens = size(initial_ens, 2)
    @assert n_ens == alg.n_ens "expected $(alg.n_ens) initial ensemble values, got $n_ens"
    # construct initial state
    state = initialstate(alg, initial_ens, obs_mean, Matrix(obs_cov); rng)
    inference_sol = SimulatorInferenceSolution(inference_prob, alg, [], [], nothing)
    return EnsembleSolver(
        inference_sol,
        alg,
        alg.ens_alg,
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
    ensemble_step!(solver)
    sol.inference_result = state
    # iteration callback
    callback_retval = solver.itercallback(state)
    # check convergence
    converged = hasconverged(alg, state)
    maxiters_reached = state.iter >= alg.maxiters
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
    return solver.sol
end

"""
    ensemble_predict(
        state::EnsembleState,
        initial_prob::SciMLBase.AbstractSciMLProblem,
        ensalg::SciMLBase.BasicEnsembleAlgorithm,
        dealg::DEAlgorithm,
        param_map::ParameterMapping,
        ensemble_prob_func,
        ensemble_output_func,
        ensemble_pred_func;
        solve_kwargs...
    )

Performs a single step/iteration for the given ensemble and returns a named tuple
`(; pred, sol)` where `sol` are the full ensemble forward solutions and `pred` is
the prediction matrix produced by `ensemble_pred_func`.
"""
function ensemble_predict(
    state::EnsembleState,
    initial_prob::SciMLBase.AbstractSciMLProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    dealg::DEAlgorithm,
    param_map::ParameterMapping,
    ensemble_prob_func,
    ensemble_output_func,
    ensemble_pred_func;
    solve_kwargs...
)
    Θ = get_ensemble(state)
    N_ens = size(Θ,2)
    # construct EnsembleProblem for forward simulations
    function prob_func(prob, i, repeat)
        ϕ = param_map(Θ[:,i])
        return ensemble_prob_func(prob, ϕ)
    end
    ensprob = EnsembleProblem(initial_prob; prob_func, output_func=(sol,i) -> ensemble_output_func(sol, i, state.iter))
    sol = solve(ensprob, dealg, ensalg; trajectories=N_ens, solve_kwargs...)
    pred = reduce(hcat, map((i,out) -> ensemble_pred_func(out, i, state.iter), 1:N_ens, sol.u))
    return (; pred, sol)
end

function default_forward_solve_pred(sol::SimulatorForwardSolution, i, iter)
    # check forward solver return code
    @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
    # retrieve observables and flatten them into a vector
    observables = sol.prob.observables
    return reduce(vcat, map(obs -> vec(retrieve(obs)), observables))
end
