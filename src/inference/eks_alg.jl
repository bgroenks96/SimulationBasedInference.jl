using EnsembleKalmanProcesses

struct EKS
    n_ens::Int # ensemble size
    ens_alg::SciMLBase.EnsembleAlgorithm # ensemble alg
    prior::MvNormal # unconstrained prior
    obs_cov::Function # obs covariance function
    output_func::Function # output function
    pred_func::Function # prediction function
    function EKS(
        n_ens::Int,
        ens_alg::SciMLBase.EnsembleAlgorithm,
        prior::MvNormal;
        obs_cov=eks_obs_cov,
        output_func=default_eks_inference_output,
        pred_func=(y,i,iter) -> y,
    )
        return new(n_ens, ens_alg, prior, obs_cov, output_func, pred_func)
    end
end

include("ekp.jl")

eks_obs_cov(likelihoods...) = error("EKS currently supports only (diagonal) Gaussian likelihoods")

# currently only diagonal covariances are supported
function eks_obs_cov(likelihoods::Union{DiagonalGaussianLikelihood,IsotropicGaussianLikelihood}...)
    cov_diags = map(likelihoods) do lik
        return diag(covariance(lik, collect(mean(lik.prior))))
    end
    # concatenate all covariance matrices 
    return Diagonal(reduce(vcat, cov_diags))
end

function default_eks_inference_output(sol::SimulatorForwardSolution, i, iter)
    @assert sol.sol.retcode ∈ (ReturnCode.Default, ReturnCode.Success) "$(sol.sol.retcode)"
    observables = sol.prob.observables
    return reduce(vcat, map(vec ∘ retrieve, observables)), false
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
    alg::EKS;
    prob_func=(prob,p) -> remake(prob, p=p),
    ekp_ctor=EnsembleKalmanProcess,
    rng=Random.default_rng(),
    initial_ens=nothing,
    maxiters=10,
    warmstart=false,
    output_dir=".",
    kwargs...
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
        samples = sample(model_prior, alg.n_ens)
        # apply transform to samples and then concatenate on second axis
        initial_ens = reduce(hcat, map(constrained_to_unconstrained, samples))
    end
    n_ens = size(initial_ens, 2)
    @assert n_ens == alg.n_ens "expected $(alg.n_ens) initial ensemble values, got $n_ens"
    # build EKP using constructor function
    ekp = ekp_ctor(initial_ens, obs_mean, Matrix(obs_cov), sampler; rng)
    forward_prob = inference_prob.forward_prob
    # fit EKP
    ekpstate = fitekp!(
        ekp,
        forward_prob,
        alg.ens_alg,
        inference_prob.forward_solver,
        ParameterMapping(inverse(constrained_to_unconstrained)),
        alg.output_func,
        alg.pred_func;
        prob_func=(forward_prob, p) -> SimulatorForwardProblem(prob_func(forward_prob.prob, p), deepcopy(forward_prob.observables)...),
        output_dir,
        maxiters,
        warmstart,
        # disable automatic saving; the forward problem integrator will handle this!
        saveat=[],
        kwargs...,
    )
    return ekpstate
end
