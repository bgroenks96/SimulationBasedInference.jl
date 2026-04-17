"""
    EKS{NF} <: EnsembleInferenceAlgorithm

Represents a proxy for the Ensemble Kalman Sampler (Garbuno-Inigo et al. 2020) implementation provided by `EnsembleKalmanProcesses`.
"""
Base.@kwdef struct EKS{NF} <: EnsembleInferenceAlgorithm
    prior_approx::GaussianApproximationMethod = LaplaceMethod()
    maxiters::Int = 30
    minΔt::NF = 2.0
end

isiterative(alg::EKS) = true

# State type for Ensemble Kalman Processes

mutable struct EKPState{NF,ekpType<:EnsembleKalmanProcess{NF}} <: EnsembleState
    ekp::ekpType
    iter::Int  # iteration step
end

get_ensemble(state::EKPState) = get_u_final(state.ekp)

get_obs_mean(state::EKPState) = get_obs(state.ekp)

get_obs_cov(state::EKPState) = get_obs_noise_cov(state.ekp)

logdensity_prior(ekp::EnsembleKalmanProcess, θ) = 0.0
logdensity_prior(ekp::EnsembleKalmanProcess{<:Sampler}, θ) = logpdf(MvNormal(ekp.process.prior_mean, ekp.process.prior_cov), θ)

hasconverged(alg::EKS, state::EKPState) = length(state.ekp.Δt) > 1 ? sum(state.ekp.Δt[2:end]) >= alg.minΔt : false

function initialstate(
    eks::EKS,
    prior::AbstractSimulatorPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obs_cov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    kwargs...
)
    unconstrained_prior = gaussian_approx(eks.prior_approx, prior; rng)
    sampler = Sampler{eltype(ens),EnsembleKalmanProcesses.EKS}(collect(mean(unconstrained_prior)), cov(unconstrained_prior))
    ekp = EnsembleKalmanProcess(ens, obs, Matrix(obs_cov), sampler; rng, kwargs...)
    return EKPState(ekp, 0)
end

function ensemblestep!(solver::EnsembleSolver{<:EKS})
    state = solver.state
    alg = solver.alg
    ekp = state.ekp
    solver.verbose && @info "Starting iteration $(state.iter) (maxiters=$(alg.maxiters))"
    Θ = get_u_final(ekp)
    # generate ensemble predictions
    out = ensemble_forward(solver)
    # update ensemble
    update_ensemble!(ekp, out.pred)
    # compute likelihoods and prior prob
    loglik = map(y -> logpdf(MvNormal(get_obs(ekp), get_obs_noise_cov(ekp)), y), eachcol(out.pred))
    logprior = map(θᵢ -> logdensity_prior(ekp, θᵢ), eachcol(Θ))
    # update ensemble solver state
    push!(solver.loglik, loglik)
    push!(solver.logprior, logprior)
    # postamble
    # calculate change in error
    err = get_error(ekp)
    Δerr = length(err) > 1 ? err[end] - err[end-1] : missing
    solver.verbose && @info "Finished iteration $(state.iter); err: $(err[end]), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
    return out
end
