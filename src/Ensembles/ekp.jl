mutable struct EKPState{ekpType<:EnsembleKalmanProcess}
    ekp::ekpType
    iter::Int  # iteration step
    loglik::Vector # log likelihoods
    logprior::Vector # log prior prob
end

hasconverged(ekp::EnsembleKalmanProcess, minΔt) = length(ekp.Δt) > 1 ? sum(ekp.Δt[2:end]) >= minΔt : false

"""
    ekpstep!(
        ekp::EnsembleKalmanProcess,
        initial_prob::SciMLBase.AbstractSciMLProblem,
        ensalg::SciMLBase.BasicEnsembleAlgorithm,
        alg::DEAlgorithm,
        param_map::ParameterMapping,
        initial_prob,
        ensemble_prob_func,
        ensemble_output_func,
        ensemble_pred_func,
        iter::Int;
        solve_kwargs...
    )

Performs a single step/iteration for the given `ekp::EnsembleKalmanProcess` and returns a vector of
log probabilities (log-likelihood or log-joint for EKS) for each ensemble member.
"""
function ekpstep!(
    ekp::EnsembleKalmanProcess,
    initial_prob::SciMLBase.AbstractSciMLProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    alg::DEAlgorithm,
    param_map::ParameterMapping,
    ensemble_prob_func,
    ensemble_output_func,
    ensemble_pred_func,
    iter::Int;
    solve_kwargs...
)
    # get current ensemeble state (unconstrained parameters)
    Θ = get_u_final(ekp)
    N_ens = size(Θ,2)
    # construct EnsembleProblem for forward simulations
    function prob_func(prob, i, repeat)
        ϕ = param_map(Θ[:,i])
        return ensemble_prob_func(prob, ϕ)
    end
    ensprob = EnsembleProblem(initial_prob; prob_func, output_func=(sol,i) -> ensemble_output_func(sol, i, iter))
    enssol = solve(ensprob, alg, ensalg; trajectories=N_ens, solve_kwargs...)
    enspred = reduce(hcat, map((i,out) -> ensemble_pred_func(out, i, iter), 1:N_ens, enssol.u))
    # update ensemble
    update_ensemble!(ekp, enspred)
    # compute likelihoods and prior prob
    loglik = map(y -> logpdf(MvNormal(y, ekp.obs_noise_cov), y), eachcol(enspred))
    logprior = map(θᵢ -> logpdf(MvNormal(ekp.process.prior_mean, ekp.process.prior_cov), θᵢ), eachcol(θ))
    return enssol, loglik, logprior
end

"""
    fitekp!(
        ekp::EnsembleKalmanProcess,
        initial_prob::SciMLBase.AbstractSciMLProblem,
        ensalg::SciMLBase.BasicEnsembleAlgorithm,
        alg::DEAlgorithm,
        param_map,
        output_func,
        ensemble_pred_func;
        maxiters=10,
        minΔt=2.0,
        initialstate=EKPState(ekp, 0, []),
        itercallback=(state, i) -> true,
        verbose=true,
        solve_kwargs...
    )

Fits an ensemble of models based on `setup` using the `ekp::EnsembleKalmanProcess` with parameter mapping function `param_map`.
The `output_func` argument is similar to that of `EnsembleProblem`; i.e. it should be a function
`(sol,i,iter)::Any` that returns and/or saves the output from the `ODESolution`. Note the additional `iter` argument which is the current
iteration index for the EKP. `ensemble_pred_func` should be a function `(out,i,iter)::AbstractVector` which takes as arguments the value
returned by `output_func`, the ensemble member identifier `i`, and the current iteration `iter`, and produces a vector of "predictions"
which must match the dimensionality of the observations provided to the `EnsembleKalmanProcess`. This vector `x` will then be passed to
[`EnsembleKalmanProcesses.update_ensemble!(ekp, x)`](@ref). Note that the optional `prob_func(prob, u0, p)` differs from the `prob_func`
passed to `EnsembleProblem` in that it directly accepts the prototype problem as well as parameters generated from the current ensemble.
The default `prob_func` implementation simply invokes `remake(prob, p=p, u0=u0)`.
"""
function fitekp!(
    ekp::EnsembleKalmanProcess,
    initial_prob::SciMLBase.AbstractSciMLProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    alg::DEAlgorithm,
    param_map,
    output_func,
    ensemble_pred_func;
    prob_func=(prob, p) -> remake(prob, p=p),
    maxiters=10,
    minΔt=2.0,
    initialstate=EKPState(ekp, 0, [], []),
    itercallback=(state, i) -> true,
    verbose=true,
    solve_kwargs...
)
    state = initialstate
    i = state.iter
    loglik = state.loglik
    logprior = state.logprior
    ekp = state.ekp
    err = i >= 1 ? ekp.err[end] : missing
    err_prev = i > 1 ? ekp.err[end-1] : missing
    Δerr = err - err_prev
    i += 1
    while !hasconverged(ekp, minΔt) && i <= maxiters
        verbose && @info "Starting iteration $i (maxiters=$(maxiters))"
        _, loglikᵢ, logpriorᵢ = ekpstep!(ekp, initial_prob, ensalg, alg, param_map, prob_func, output_func, ensemble_pred_func, i; solve_kwargs...)
        push!(loglik, loglikᵢ)
        push!(logprior, logpriorᵢ)
        err = ekp.err[end]
        Δerr = length(ekp.err) > 1 ? err - ekp.err[end-1] : missing
        state.iter = i
        verbose && @info "Finished iteration $i; err: $(err), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
        if !itercallback(state, i)
            # terminate iteration if itercallback returns false
            break
        end
        i += 1
    end
    return state
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
