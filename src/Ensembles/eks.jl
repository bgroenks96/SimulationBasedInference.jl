using EnsembleKalmanProcesses

"""
    EKS <: EnsembleInferenceAlgorithm

Represents a proxy for the Ensemble Kalman Sampler implementation provided by `EnsembleKalmanProcesses`.
"""
Base.@kwdef struct EKS <: EnsembleInferenceAlgorithm
    n_ens::Int # ensemble size
    ens_alg::SciMLBase.EnsembleAlgorithm # ensemble alg
    prior::MvNormal # unconstrained prior
    obs_cov::Function = obscov # obs covariance function
    maxiters::Int = 30
    minΔt::Float64 = 2.0
end
# Additional constructors
EKS(n_ens::Int, ens_alg::SciMLBase.EnsembleAlgorithm, prior::MvNormal; kwargs...) = EKS(; n_ens, ens_alg, prior, kwargs...)
function EKS(
    n_ens::Int,
    ens_alg::SciMLBase.EnsembleAlgorithm,
    prior::AbstractPrior,
    rng::Random.AbstractRNG=Random.GLOBAL_RNG;
    num_prior_samples=10_000,
    kwargs...
)
    constrained_to_unconstrained = bijector(prior)
    constrained_prior_samples = sample(rng, prior, num_prior_samples)
    unconstrained_prior_samples = reduce(hcat, map(constrained_to_unconstrained, constrained_prior_samples))
    unconstrained_mean = mean(unconstrained_prior_samples, dims=2)
    unconstrained_var = var(unconstrained_prior_samples, dims=2)
    eks_prior = MvNormal(unconstrained_mean[:,1], Diagonal(unconstrained_var[:,1]))
    return EKS(n_ens, ens_alg, eks_prior; kwargs...)
end

hasconverged(alg::EKS, state::EKPState) = length(state.ekp.Δt) > 1 ? sum(state.ekp.Δt[2:end]) >= alg.minΔt : false

function initialstate(
    eks::EKS,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obscov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    kwargs...
)
    sampler = Sampler(mean(eks.prior), cov(eks.prior))
    ekp = EnsembleKalmanProcess(ens, obs, Matrix(obscov), sampler; rng, kwargs...)
    return EKPState(ekp, 0, [], [])
end

function ensemble_step!(solver::EnsembleSolver{EKS})
    sol = solver.sol
    state = solver.state
    alg = solver.alg
    ekp = state.ekp
    solver.verbose && @info "Starting iteration $(state.iter) (maxiters=$(alg.maxiters))"
    # parameter mapping (model parameters only)
    param_map = ParameterMapping(sol.prob.prior.model)
    # generate ensemble predictions
    enspred, _ = ensemble_predict(
        state,
        sol.prob.forward_prob,
        solver.ensalg,
        sol.prob.forward_solver,
        param_map,
        solver.prob_func,
        solver.output_func,
        solver.pred_func;
        solver.solve_kwargs...
    )
    # update ensemble
    update_ensemble!(ekp, enspred)
    # compute likelihoods and prior prob
    Θ = get_u_final(ekp)
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
