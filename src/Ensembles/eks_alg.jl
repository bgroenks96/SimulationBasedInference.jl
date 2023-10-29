using EnsembleKalmanProcesses

"""
    EKS <: SimulatorInferenceAlgorithm

Represents a proxy for the Ensemble Kalman Sampler implementation provided by `EnsembleKalmanProcesses`.
"""
Base.@kwdef struct EKS <: SimulatorInferenceAlgorithm
    n_ens::Int # ensemble size
    ens_alg::SciMLBase.EnsembleAlgorithm # ensemble alg
    prior::MvNormal # unconstrained prior
    obs_cov::Function = eks_obs_cov # obs covariance function
    maxiters::Int = 30
    minΔt::Float64 = 2.0
end

EKS(n_ens::Int, ens_alg::SciMLBase.EnsembleAlgorithm, prior::MvNormal; kwargs...) = EKS(; n_ens, ens_alg, prior, kwargs...)

function EKS(
    n_ens::Int,
    ens_alg::SciMLBase.EnsembleAlgorithm,
    prior::AbstractPrior,
    rng::Random.AbstractRNG=Random.GLOBAL_RNG;
    num_prior_samples=1000,
    kwargs...
)
    constrained_to_unconstrained = bijector(prior)
    prior_samples = reduce(hcat, map(constrained_to_unconstrained, sample(rng, prior, num_prior_samples)))
    unconstrained_mean = mean(prior_samples, dims=2)
    unconstrained_var = var(prior_samples, dims=2)
    eks_prior = MvNormal(unconstrained_mean[:,1], Diagonal(unconstrained_var[:,1]))
    return EKS(n_ens, ens_alg, eks_prior; kwargs...)
end

"""
    init(::SimulatorInferenceProblem, ::EKS, ensemble_alg; kwargs...)

Initializes EKS for the given `SimulatorInferenceProblem` using the Ensemble Kalman Sampling algorithm. This method automatically
constructs an `EnsembleKalmanProcess` from the given inference problem and `named_data` pairs. The `ekp_ctor` keyword
argument allows the user to modify the construction of the `EnsembleKalmanProcess` (default is to invoke the constructor
directly). If `initial_ens` is not provided, the initial ensemble is sampled from the prior.
"""
function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::EKS;
    prob_func=(prob,p) -> remake(prob, p=p),
    output_func=(sol,i,iter) -> (sol, false),
    pred_func=default_forward_solve_pred,
    ekp_ctor=EnsembleKalmanProcess,
    rng=Random.default_rng(),
    initial_ens=nothing,
    itercallback=(state, i) -> true,
    verbose=true,
    solve_kwargs...
)
    # extract model prior (i.e. ignoring likelihood parameters)
    model_prior = inference_prob.prior.model
    # construct EKS
    sampler = Sampler(mean(alg.prior), Matrix(cov(alg.prior)))
    # extract observations from likelihood terms
    names = keys(inference_prob.data)
    data = values(inference_prob.data)
    likelihoods = map(n -> getproperty(inference_prob.likelihoods, n), names)
    obs_mean = reduce(vcat, map(vec, data))
    obs_cov = alg.obs_cov(likelihoods...)
    # construct transform from model prior
    constrained_to_unconstrained = bijector(inference_prob.prior.model)
    # sample initial ensemble
    if isnothing(initial_ens)
        samples = sample(rng, model_prior, alg.n_ens)
        # apply transform to samples and then concatenate on second axis
        initial_ens = reduce(hcat, map(constrained_to_unconstrained, samples))
    end
    n_ens = size(initial_ens, 2)
    @assert n_ens == alg.n_ens "expected $(alg.n_ens) initial ensemble values, got $n_ens"
    # build EKP using constructor function
    ekp = ekp_ctor(initial_ens, obs_mean, Matrix(obs_cov), sampler; rng)
    ekpstate = EKPState(ekp, 0, [])
    inference_sol = SimulatorInferenceSolution(inference_prob, [], nothing)
    return EnsembleSolver(
        inference_sol,
        alg,
        alg.ens_alg,
        ekpstate,
        prob_func,
        output_func,
        pred_func,
        itercallback,
        verbose,
        ReturnCode.Default,
        solve_kwargs,
    )
end

function CommonSolve.step!(solver::EnsembleSolver{<:SimulatorInferenceProblem,EKS})
    # if retcode is already set to a terminal value, return immediately
    if solver.retcode != ReturnCode.Default
        return false
    end
    alg = solver.alg
    sol = solver.sol
    inference_prob = sol.prob
    state = solver.state
    ekp = state.ekp
    state.iter += 1
    solver.verbose && @info "Starting iteration $(state.iter) (maxiters=$(alg.maxiters))"
    # parameter mapping (model parameters only)
    constrained_to_unconstrained = bijector(inference_prob.prior.model)
    param_map = ParameterMapping(inverse(constrained_to_unconstrained))
    # EKS iteration
    enssol, logprobsᵢ = ekpstep!(
        ekp,
        inference_prob.forward_prob,
        alg.ens_alg,
        inference_prob.forward_solver,
        param_map,
        solver.prob_func,
        solver.output_func,
        solver.pred_func,
        state.iter;
        solver.solve_kwargs...
    )
    # update ensemble solver state
    push!(state.lp, logprobsᵢ)
    push!(sol.sols, enssol)
    sol.inference_result = state
    # calculate change in error
    err = ekp.err[end]
    Δerr = length(ekp.err) > 1 ? err - ekp.err[end-1] : missing
    solver.verbose && @info "Finished iteration $(state.iter); err: $(err), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
    # iteration callback
    callback_retval = solver.itercallback(state, state.iter)
    # check convergence
    converged = hasconverged(ekp, alg.minΔt)
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

function CommonSolve.solve!(solver::EnsembleSolver{<:SimulatorInferenceProblem,EKS})
    # step until convergence
    while CommonSolve.step!(solver) end
    return solver.sol
end

eks_obs_cov(likelihoods...) = error("EKS currently supports only (diagonal) Gaussian likelihoods")

# currently only diagonal covariances are supported
function eks_obs_cov(likelihoods::Union{DiagonalGaussianLikelihood,IsotropicGaussianLikelihood}...)
    cov_diags = map(likelihoods) do lik
        return diag(SimulationBasedInference.covariance(lik, collect(mean(lik.prior))))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end

function default_forward_solve_pred(sol::SimulatorForwardSolution, i, iter)
    @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
    observables = sol.prob.observables
    # retrieve observables and flatten them into a vector
    return reduce(vcat, map(obs -> vec(retrieve(obs)), observables))
end
