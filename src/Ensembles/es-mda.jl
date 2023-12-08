mutable struct ESMDAState{ensType,meanType,covType}
    ens::Vector{ensType}
    pred::Vector{ensType}
    obs_mean::meanType
    obs_cov::covType
    loglik::Vector # log likelihoods
    logprior::Vector # log prior prob
    iter::Int  # iteration step
    rng::AbstractRNG
end

Base.@kwdef struct ESMDA
    n_ens::Int # ensemble size
    ens_alg::SciMLBase.EnsembleAlgorithm # ensemble alg
    prior::MvNormal # unconstrained prior
    obs_cov::Function = obscov # obs covariance function
    maxiters::Int = 30
    alpha::Float64 = 1.0
    ρ_AB::Float64 = 1.0
    ρ_BB::Float64 = 1.0
    stochastic::Bool = true
    dosvd::Bool = true
    svd_thresh::Float64 = 0.90
end

getensemble(state::ESMDAState) = state.ens[end]

function initialstate(
    esmda::ESMDA,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obs_cov::AbstractMatrix;
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    @assert esmda.n_ens == size(ens,2) "ensemble sizes do not match"
    # initially empty array of prediction matrices
    preds = typeof(ens)[]
    return ESMDAState([ens], preds, obs, obs_cov, [], [], 0, rng)
end

function ensemble_step!(solver::EnsembleSolver{<:ESMDA})
    sol = solver.sol
    state = solver.state
    alg = solver.alg
    rng = state.rng
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
    # Kalman update
    @unpack ρ_AB, ρ_BB, stochastic, dosvd, svd_thresh = alg
    ensemble_kalman_analysis(
        Θ,
        state.obs_mean,
        enspred,
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
    loglik = map(y -> logpdf(MvNormal(state.obs_mean, state.obs_noise_cov), y), eachcol(enspred))
    logprior = map(θᵢ -> logpdf(alg.prior, θᵢ), eachcol(Θ))
    # update ensemble solver state
    push!(state.loglik, loglik)
    push!(state.logprior, logprior)
    push!(sol.inputs, Θ)
    push!(sol.outputs, enspred)
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
    n_obs, n_ens = size(pred)
    n_par = size(prior, 1)
    prior = reshape(prior, n_par, n_ens)
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
    aR = n_ens*alpha*R

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
        ϵ = randn(rng, n_obs, n_ens)
        Rsqrt = sqrt(R)
        Y = obs*ones(n_ens)' + sqrt(alpha)*Rsqrt*ϵ
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
