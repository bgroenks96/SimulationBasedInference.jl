# State type for Ensemble Kalman Processes

mutable struct EKPState{ekpType<:EnsembleKalmanProcess} <: EnsembleState
    ekp::ekpType
    iter::Int  # iteration step
    loglik::Vector # log likelihoods
    logprior::Vector # log prior prob
end

get_ensemble(state::EKPState) = get_u_final(state.ekp)

get_obs_mean(state::EKPState) = state.ekp.obs_mean

get_obs_cov(state::EKPState) = state.ekp.obs_noise_cov

logdensity_prior(ekp::EnsembleKalmanProcess, θ) = 0.0
logdensity_prior(ekp::EnsembleKalmanProcess{<:Sampler}, θ) = logpdf(MvNormal(ekp.process.prior_mean, ekp.process.prior_cov), θ)

"""
    EKS <: EnsembleInferenceAlgorithm

Represents a proxy for the Ensemble Kalman Sampler implementation provided by `EnsembleKalmanProcesses`.
"""
Base.@kwdef struct EKS <: EnsembleInferenceAlgorithm
    prior_approx::GaussianApproximationMethod = LaplaceMethod()
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

function ensemblestep!(solver::EnsembleSolver{EKS})
    sol = solver.sol
    state = solver.state
    alg = solver.alg
    ekp = state.ekp
    solver.verbose && @info "Starting iteration $(state.iter) (maxiters=$(alg.maxiters))"
    Θ = get_u_final(ekp)
    # parameter mapping (model parameters only)
    param_map = ParameterTransform(sol.prob.prior.model)
    # generate ensemble predictions
    out = ensemble_solve(
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
    update_ensemble!(ekp, out.pred)
    # compute likelihoods and prior prob
    loglik = map(y -> logpdf(MvNormal(ekp.obs_mean, ekp.obs_noise_cov), y), eachcol(out.pred))
    logprior = map(θᵢ -> logdensity_prior(ekp, θᵢ), eachcol(Θ))
    # update ensemble solver state
    push!(state.loglik, loglik)
    push!(state.logprior, logprior)
    store!(sol.storage, Θ, out.observables, iter=state.iter)
    # postamble
    # calculate change in error
    err = ekp.err[end]
    Δerr = length(ekp.err) > 1 ? err - ekp.err[end-1] : missing
    solver.verbose && @info "Finished iteration $(state.iter); err: $(err), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
end

"""
    Chains(ekp::EnsembleKalmanProcess, parameter_names::AbstractVector{Symbol}, iter=nothing)

Constructs a `Chains` object from the given (fitted) EKP by mapping the unconstrained parameters
at iteration `iter` (defaulting to the last iteration) back to constrained space.
"""
function MCMCChains.Chains(ekp::EnsembleKalmanProcess, parameter_names::AbstractVector{Symbol}, iter=nothing)
    # N_p x N_ens
    Θ = isnothing(iter) ? get_u_final(ekp) : get_u(ekp, iter)
    Φ = transpose(reduce(hcat, map(inverse(bijector(prior.model)), eachcol(Θ))))
    return MCMCChains.Chains(Φ, parameter_names)
end
