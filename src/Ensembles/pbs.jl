"""
    PBS

Basic particle batch smoother that is effectively quivalent to
naive importance sampling of the posterior.
"""
Base.@kwdef struct PBS <: EnsembleInferenceAlgorithm
    obs_cov::Function = obscov # obs covariance function
end

mutable struct PBSState{ensType,meanType,covType} <: EnsembleState
    ens::ensType
    obs_mean::meanType
    obs_cov::covType
    loglik::Vector{Float64} # log likelihoods
    weights::Vector{Float64} # importance weights
    Neff::Int # effective sample size
    iter::Int  # iteration step
end

get_ensemble(state::PBSState) = state.ens

get_obs_mean(state::PBSState) = state.obs_mean

get_obs_cov(state::PBSState) = state.obs_cov

get_weights(state::PBSState) = state.weights

function initialstate(
    pbs::PBS,
    prior::AbstractPrior,
    ens::AbstractMatrix,
    obs::AbstractVector,
    obs_cov::AbstractMatrix;
    rng::AbstractRNG=Random.default_rng(),
)
    return PBSState(ens, obs, obs_cov, Float64[], Float64[], -1, 0)
end

function ensemblestep!(solver::EnsembleSolver{<:PBS})
    sol = solver.sol
    state = solver.state
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
    # compute importance weights
    w, Neff = importance_weights(state.obs_mean, enspred, state.obs_cov)
    # compute likelihoods
    loglik = map(y -> logpdf(MvNormal(state.obs_mean, state.obs_cov), y), eachcol(enspred))
    state.weights = w
    state.Neff = Neff
    state.loglik = loglik
    push!(sol.inputs, state.ens)
    push!(sol.outputs, enspred)
end

function ensemble_predict(solver::EnsembleSolver{<:PBS})
    if solver.state.iter > 0
        enspred = solver.sol.outputs[end]
    else
        enspred, _ = ensemble_solve(
            solver.state,
            solver.sol.prob.forward_prob,
            solver.ensalg,
            solver.sol.prob.forward_solver,
            ParameterTransform(solver.sol.prob.prior.model);
            prob_func=solver.prob_func,
            output_func=solver.output_func,
            pred_func=solver.pred_func,
            solver.solve_kwargs...
        )
    end
    return enspred
end
