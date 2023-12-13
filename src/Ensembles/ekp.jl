mutable struct EKPState{ekpType<:EnsembleKalmanProcess} <: EnsembleState
    ekp::ekpType
    iter::Int  # iteration step
    loglik::Vector # log likelihoods
    logprior::Vector # log prior prob
end

getensemble(state::EKPState) = get_u_final(state.ekp)

get_obs_mean(state::EKPState) = state.ekp.obs_mean

get_obs_cov(state::EKPState) = state.ekp.obs_noise_cov

logdensity_prior(ekp::EnsembleKalmanProcess, θ) = 0.0
logdensity_prior(ekp::EnsembleKalmanProcess{<:Sampler}, θ) = logpdf(MvNormal(ekp.process.prior_mean, ekp.process.prior_cov), θ)

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
