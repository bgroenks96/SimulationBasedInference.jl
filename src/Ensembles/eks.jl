"""
    EKS <: EnsembleInferenceAlgorithm

Represents a proxy for the Ensemble Kalman Sampler implementation provided by `EnsembleKalmanProcesses`.
"""
Base.@kwdef struct EKS <: EnsembleInferenceAlgorithm
    prior_approx::GaussianApproximationMethod = EmpiricalGaussian()
    obs_cov::Function = obscov # obs covariance function
    maxiters::Int = 30
    minΔt::Float64 = 2.0
end

isiterative(alg::EKS) = true

hasconverged(alg::EKS, state::EKPState) = length(state.ekp.Δt) > 1 ? sum(state.ekp.Δt[2:end]) >= alg.minΔt : false

function initialstate(
    eks::EKS,
    prior::AbstractPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obs_cov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    kwargs...
)
    unconstrained_prior = gaussian_approx(eks.prior_approx, prior; rng)
    sampler = Sampler(mean(unconstrained_prior), cov(unconstrained_prior))
    ekp = EnsembleKalmanProcess(ens, obs, Matrix(obs_cov), sampler; rng, kwargs...)
    return EKPState(ekp, 0, [], [])
end

function ensemble_step!(solver::EnsembleSolver{EKS})
    sol = solver.sol
    state = solver.state
    alg = solver.alg
    ekp = state.ekp
    solver.verbose && @info "Starting iteration $(state.iter) (maxiters=$(alg.maxiters))"
    Θ = get_u_final(ekp)
    # parameter mapping (model parameters only)
    param_map = ParameterTransform(sol.prob.prior.model)
    # generate ensemble predictions
    enspred, _ = ensemble_solve(
        state,
        sol.prob.forward_prob,
        solver.ensalg,
        sol.prob.forward_solver,
        param_map;
        prob_func=solver.prob_func,
        output_func=solver.output_func,
        pred_func=solver.pred_func,
        solver.solve_kwargs...
    )
    # update ensemble
    update_ensemble!(ekp, enspred)
    # compute likelihoods and prior prob
    loglik = map(y -> logpdf(MvNormal(ekp.obs_mean, ekp.obs_noise_cov), y), eachcol(enspred))
    logprior = map(θᵢ -> logdensity_prior(ekp, θᵢ), eachcol(Θ))
    # update ensemble solver state
    push!(state.loglik, loglik)
    push!(state.logprior, logprior)
    push!(sol.inputs, Θ)
    push!(sol.outputs, enspred)
    # postamble
    # calculate change in error
    err = ekp.err[end]
    Δerr = length(ekp.err) > 1 ? err - ekp.err[end-1] : missing
    solver.verbose && @info "Finished iteration $(state.iter); err: $(err), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
end
