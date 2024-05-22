"""
    EnIS

Basic ensemble importance sampling (EnIS) inference algorithm.
Also sometimes referred to as a "particle batch smoother" (PBS) or
"generalized likelihood uncertainty estimation" (GLUE) depending on
the context.
"""
Base.@kwdef struct EnIS <: EnsembleInferenceAlgorithm end

"""
Alias for EnIS.
"""
const PBS = EnIS

mutable struct EnISState{ensType,meanType,covType} <: EnsembleState
    ens::ensType
    obs_mean::meanType
    obs_cov::covType
    loglik::Vector{Float64} # log likelihoods
    weights::Vector{Float64} # importance weights
    Neff::Int # effective sample size
    iter::Int  # iteration step
end

get_ensemble(state::EnISState) = state.ens

get_obs_mean(state::EnISState) = state.obs_mean

get_obs_cov(state::EnISState) = state.obs_cov

get_weights(state::EnISState) = state.weights

get_weights(sol::SimulatorInferenceSolution{EnIS}) = get_weights(sol.result)

function initialstate(
    pbs::EnIS,
    prior::AbstractPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obs_cov::AbstractMatrix;
    rng::AbstractRNG=Random.default_rng(),
)
    return EnISState(ens, obs, obs_cov, Float64[], Float64[], -1, 0)
end

function ensemblestep!(solver::EnsembleSolver{<:EnIS})
    sol = solver.sol
    state = solver.state
    # model parameter forward map
    param_map = unconstrained_forward_map(sol.prob.prior.model)
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
    # compute importance weights
    w, Neff = importance_weights(state.obs_mean, out.pred, state.obs_cov)
    # compute likelihoods
    loglik = map(y -> logpdf(MvNormal(state.obs_mean, state.obs_cov), y), eachcol(out.pred))
    state.weights = w
    state.Neff = Neff
    state.loglik = loglik
    store!(sol.storage, state.ens, out.observables, iter=state.iter)
end

finalize!(::EnsembleSolver{EnIS}) = nothing

"""
    importance_weights(obs::AbstractVector, pred::AbstractMatrix, R::AbstractMatrix)

Computes the importance weights from the given observations, predictions, and noise covariance.
When used for data assimilation, this algorithm is also known as the "particle batch smoother".
This implementation is adapted from the python implementation by Kristoffer Aalstad:

https://github.com/ealonsogzl/MuSA/blob/0c02b8dc25a0f902bf63099de68174f4738705f0/modules/filters.py#L169

Original docstring:

Inputs:
    obs: Observation vector (m x 1 array)
    pred: Predicted observation ensemble matrix (m x N array)
    R: Observation error covariance 'matrix' (m x 1 array, or scalar)
Outputs:
    w: Posterior weights (N x 1 array)
    Neff: Effective sample size
Dimensions:
    N is the number of ensemble members and m is the number
    of observations.

Here we have implemented the particle batch smoother, which is
a batch-smoother version of the particle filter (i.e. a particle filter
without resampling), described in Margulis et al.
(2015, doi: 10.1175/JHM-D-14-0177.1). As such, this routine can also be
used for particle filtering with sequential data assimilation. This scheme
is obtained by using a particle (mixture of Dirac delta functions)
representation of the prior and applying this directly to Bayes theorem. In
other words, it is just an application of importance sampling with the
prior as the proposal density (importance distribution). It is also the
same as the Generalized Likelihood Uncertainty Estimation (GLUE) technique
(with a formal Gaussian likelihood)which is widely used in hydrology.

This particular scheme assumes that the observation errors are additive
Gaussian white noise (uncorrelated in time and space). The "logsumexp"
trick is used to deal with potential numerical issues with floating point
operations that occur when dealing with ratios of likelihoods that vary by
many orders of magnitude.
"""
function importance_weights(obs::AbstractVector, pred::AbstractMatrix, R_cov)
    n_obs = size(pred, 1)
    R = obscov(R_cov)*Diagonal(ones(n_obs))
    return importance_weights(obs, pred, R)
end

function importance_weights(obs::AbstractVector, pred::AbstractMatrix, R::Diagonal)
    n_obs, ensemble_size = size(pred)
    @assert n_obs == length(obs)
    residual = obs .- pred
    loglik = sum(-0.5*inv(R)*residual.^2, dims=1)[1,:]

    # Log of normalizing constant
    log_z = logsumexp(loglik)

    # Weights
    logw = loglik .- log_z
    weights = exp.(logw)

    # sanity checks
    @assert length(weights) == ensemble_size "weight vector has incorrect dimensions: $(size(weights))"
    @assert abs(sum(weights) - 1.0) < sqrt(eps()) "particle weights do not sum to unity! ∑w=$(sum(weights))"

    Neff = round(1/sum(weights.^2))

    return weights, Neff
end

