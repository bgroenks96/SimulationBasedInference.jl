"""
    ESMDA{NF} <: EnsembleInferenceAlgorithm

Implementation of the ensemble-smother multiple data assimilation algorithm of Emerick et al. 2013.

Emerick, Alexandre A., and Albert C. Reynolds. "Ensemble smoother with multiple data assimilation." Computers & Geosciences 55 (2013): 3-15.
"""
Base.@kwdef struct ESMDA{NF} <: EnsembleInferenceAlgorithm
    prior_approx::GaussianApproximationMethod = LaplaceMethod()
    maxiters::Int = 4
    alpha::NF = maxiters
    ρ_AB::NF = 1.0
    ρ_BB::NF = 1.0
    stochastic::Bool = true
    dosvd::Bool = true
    svd_thresh::NF = 0.90
end

mutable struct ESMDAState{
    NF,
    ensType<:AbstractMatrix{NF},
    meanType,
    covType
} <: EnsembleState
    ens::ensType
    obs_mean::meanType
    obs_cov::covType
    prior::MvNormal
    loglik::Vector{NF} # log likelihoods
    logprior::Vector{NF} # log prior prob
    iter::Int  # iteration step
    rng::AbstractRNG
end

isiterative(alg::ESMDA) = true

get_ensemble(state::ESMDAState) = state.ens

get_obs_mean(state::ESMDAState) = state.obs_mean

get_obs_cov(state::ESMDAState) = state.obs_cov

hasconverged(alg::ESMDA, state::ESMDAState) = state.iter >= alg.maxiters

function initialstate(
    esmda::ESMDA{NF},
    prior::AbstractSimulatorPrior,
    ens::AbstractMatrix{NF},
    obs::AbstractVector{NF},
    obs_cov::AbstractMatrix{NF};
    rng::AbstractRNG=Random.default_rng(),
) where {NF}
    unconstrained_prior = gaussian_approx(esmda.prior_approx, prior; rng)
    return ESMDAState(ens, obs, obs_cov, unconstrained_prior, NF[], NF[], 0, rng)
end

function ensemblestep!(solver::EnsembleSolver{<:ESMDA})
    state = solver.state
    alg = solver.alg
    rng = state.rng
    # generate ensemble predictions
    out = ensemble_forward(solver)
    # Kalman update
    (; ρ_AB, ρ_BB, stochastic, dosvd, svd_thresh) = alg
    Θ = state.ens
    Θ_post = ensemble_kalman_analysis(
        Θ,
        state.obs_mean,
        out.pred,
        alg.alpha,
        state.obs_cov;
        rng,
        ρ_AB,
        ρ_BB,
        stochastic,
        dosvd,
        svd_thresh
    )
    # compute likelihoods and prior prob
    loglik = map(y -> logpdf(MvNormal(state.obs_mean, state.obs_cov), y), eachcol(out.pred))
    logprior = map(θᵢ -> logpdf(state.prior, θᵢ), eachcol(Θ))
    # update ensemble solver and solution state
    state.ens = Θ_post
    push!(state.loglik, loglik)
    push!(state.logprior, logprior)
    return out
end

"""
    ensemble_kalman_analysis(
        prior::AbstractVecOrMat,
        obs::AbstractVector,
        pred::AbstractMatrix,
        alpha,
        R_cov;
        ρ_AB=1.0,
        ρ_BB=1.0,
        stochastic=true,
        dosvd=true,
        svd_thresh=0.90,
        rng::AbstractRNG=Random.GLOBAL_RNG,
    )

Performs a single ensemble Kalman analysis step. Adapted from the python
implementation by Kristoffer Aalstad:

https://github.com/ealonsogzl/MuSA/blob/0c02b8dc25a0f902bf63099de68174f4738705f0/modules/filters.py#L33
"""
function ensemble_kalman_analysis(
    prior::AbstractVecOrMat,
    obs::AbstractVector,
    pred::AbstractMatrix,
    alpha,
    R_cov;
    ρ_AB=1.0,
    ρ_BB=1.0,
    stochastic=true,
    dosvd=true,
    svd_thresh=0.90,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    n_obs, ensemble_size = size(pred)
    n_par = size(prior, 1)
    prior = reshape(prior, n_par, ensemble_size)
    # n_obs x n_obs
    R = obscov(R_cov)*Diagonal(ones(n_obs))

    # n_par x 1
    prior_mean = mean(prior, dims=2)
    A = prior .- prior_mean
    # n_obs x 1
    pred_mean = mean(pred, dims=2)
    B = pred .- pred_mean
    Bt = B'

    C_AB = ρ_AB*A*Bt
    C_BB = ρ_BB*B*Bt
    aR = ensemble_size*alpha*R

    if dosvd
        L = cholesky(aR).L
        Linv = inv(L)
        Ctilde = Linv*C_BB*Linv' + I
        U, S, Vt = svd(Ctilde)
        Svr = cumsum(S) / sum(S)
        trunc_idx = findfirst(Svr .> svd_thresh)
        S_trunc = Diagonal(S[1:trunc_idx])
        U_trunc = U[:,1:trunc_idx]
        Ctilde_inv = U_trunc*inv(S_trunc)*U_trunc'
        Cinv = Linv'*Ctilde_inv*Linv
    else
        Cinv = pinv(C_BB+aR)
    end

    if stochastic
        ϵ = randn(rng, n_obs, ensemble_size)
        Rsqrt = sqrt(R)
        Y = obs*ones(ensemble_size)' + sqrt(alpha)*Rsqrt*ϵ
        # Kalman gain
        K = C_AB*Cinv
        # innovation
        inno = Y - pred
        post = prior + K*inno
    else
        Y = obs
        K = C_AB*Cinv
        inno = Y - pred_mean
        posterior_mean = prior_mean + K*inno
        A = A - 0.5*K*B
        post = posterior_mean + A
    end
    return post
end
