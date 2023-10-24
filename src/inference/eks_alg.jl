using EnsembleKalmanProcesses

struct EKS
    n_ens::Int # ensemble size
    prior::MvNormal # unconstrained prior
    obs_cov::Function
    output_func::Function
    pred_func::Function
    function EKS(
        n_ens::Int,
        prior::MvNormal;
        obs_cov=eks_obs_cov,
        output_func=default_eks_inference_output,
        pred_func=(y,i,iter) -> y,
    )
        return new(n_ens, prior, obs_cov, output_func, pred_func)
    end
end

include("ekp.jl")

eks_obs_cov(likelihoods...) = error("EKS currently supports only (diagonal) Gaussian likelihoods")

# currently only diagonal covariances are supported
function eks_obs_cov(likelihoods::Union{DiagonalGaussianLikelihood,IsotropicGaussianLikelihood}...)
    cov_diags = map(likelihoods) do lik
        return diag(covariance(lik, mean(lik.prior)))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end

function default_eks_inference_output(sol::SimulatorForwardSolution, i, iter)
    @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
    observables = sol.prob.observables
    return reduce(vcat, map(vec ∘ observe, observables)), false
end

"""
    solve(::SimulatorInferenceProblem, ::EKS, ensemble_alg; kwargs...)

Solve the given `SimulatorInferenceProblem` using the Ensemble Kalman Sampling algorithm. This method automatically
constructs an `EnsembleKalmanProcess` from the given inference problem and `named_data` pairs. The `ekp_ctor` keyword
argument allows the user to modify the construction of the `EnsembleKalmanProcess` (default is to invoke the constructor
directly). If `initial_ens` is not provided, the initial ensemble is sampled from the prior.
"""
function CommonSolve.solve(
    inference_prob::SimulatorInferenceProblem,
    alg::EKS,
    ensemble_alg;
    prob_func=(prob,p) -> remake(prob, p=p),
    ekp_ctor=EnsembleKalmanProcess,
    rng=Random.default_rng(),
    initial_ens=nothing,
    maxiters=10,
    warmstart=false,
    output_dir=".",
    kwargs...
)
    true_prior = inference_prob.prior
    sampler = Sampler(mean(alg.prior), Matrix(cov(alg.prior)))
    names = keys(inference_prob.data)
    data = values(inference_prob.data)
    likelihoods = map(n -> getproperty(inference_prob.likelihoods, n), names)
    obs_mean = reduce(vcat, map(vec, data))
    obs_cov = alg.obs_cov(likelihoods...)
    if isnothing(initial_ens)
        Φ = sample(true_prior.model, Prior(), alg.n_ens, verbose=false, progress=false)
        initial_ens = reduce(hcat, map(bijector(true_prior.model), eachrow(Array(Φ))))
    end
    n_ens = size(initial_ens, 2)
    @assert n_ens == alg.n_ens "expected $(alg.n_ens) initial ensemble values, got $n_ens"
    ekp = ekp_ctor(initial_ens, obs_mean, Matrix(obs_cov), sampler; rng)
    forward_prob = inference_prob.forward_prob
    tile = Tile(forward_prob.prob.f)
    # ensemble setup with built-in saving disabled
    param_map = ParameterMapping(inference_prob.prior)
    ekpstate = fitekp!(
        ekp,
        forward_prob,
        ensemble_alg,
        inference_prob.forward_solver,
        param_map,
        alg.output_func,
        alg.pred_func;
        prob_func=(forward_prob, p) -> SimulatorForwardProblem(prob_func(prob, p), deepcopy(forward_prob.observables)...),
        output_dir,
        maxiters,
        warmstart,
        saveat=[],
        kwargs...,
    )
    return ekpstate
end
