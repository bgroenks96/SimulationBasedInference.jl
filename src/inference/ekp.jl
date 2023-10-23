"""
    fitekp!(
        ekp::EnsembleKalmanProcess,
        initial_prob::SciMLBase.DEProblem,
        ensalg::SciMLBase.BasicEnsembleAlgorithm,
        alg::DEAlgorithm,
        param_map,
        output_func,
        ensemble_pred_func;
        maxiters=10,
        maxΔt=2.0,
        warmstart=true,
        output_dir=".",
        statefile="ekpstate.jld2",
        itercallback=(i,state) -> true,
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
    initial_prob::SciMLBase.DEProblem,
    ensalg::SciMLBase.BasicEnsembleAlgorithm,
    alg::DEAlgorithm,
    param_map,
    output_func,
    ensemble_pred_func;
    prob_func=(prob, p) -> remake(prob, p=p),
    maxiters=10,
    maxΔt=2.0,
    warmstart=true,
    output_dir=".",
    statefile="ekpstate.jld2",
    itercallback=(i,state) -> true,
    solve_kwargs...
)
    hasconverged(ekp) = length(ekp.Δt) > 1 ? sum(ekp.Δt[2:end]) >= maxΔt : false
    err = missing
    err_prev = missing
    Δerr = missing
    i = 1
    logprobs = []
    statefilepath = isnothing(statefile) ? "" : joinpath(output_dir, statefile)
    # warm start if saved state file is present
    if warmstart && isfile(statefilepath)
        warmstate = load(statefilepath)
        i = warmstate["i"]
        logprobs = warmstate["lp"]
        ekp = warmstate["ekp"]
        err = ekp.err[end]
        err_prev = i > 1 ? ekp.err[end-1] : missing
        Δerr = err - err_prev
        i += 1
    end
    state = Dict("ekp" => ekp, "i" => i, "lp" => logprobs)
    while !hasconverged(ekp) && i <= maxiters
        @info "Starting iteration $i (maxiters=$(maxiters))"
        logprobsᵢ, _ = ekpstep!(ekp, initial_prob, ensalg, alg, param_map, prob_func, output_func, ensemble_pred_func, i; solve_kwargs...)
        push!(logprobs, logprobsᵢ)
        err_prev = err
        err = ekp.err[end]
        Δerr = length(ekp.err) > 1 ? err - ekp.err[end-1] : missing
        state["i"] = i
        if !isnothing(statefile)
            @info "Saving ensemble state"
            save(statefilepath, state)
        end
        @info "Finished iteration $i; err: $(err), Δerr: $Δerr, Δt: $(sum(ekp.Δt[2:end]))"
        if !itercallback(i, state)
            # terminate iteration if itercallback returns false
            break
        end
        i += 1
    end
    return state
end
"""
    ekpstep!(
        ekp::EnsembleKalmanProcess,
        initial_prob::SciMLBase.DEProblem,
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
    initial_prob::SciMLBase.DEProblem,
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
    ensprob = EnsembleProblem(initial_prob; prob_func, output_func=(sol,i) -> ensemble_output_func(sol,i,iter))
    enssol = solve(ensprob, alg, ensalg; trajectories=N_ens, solve_kwargs...)
    enspred = reduce(hcat, map((i,out) -> ensemble_pred_func(out, i, iter), 1:N_ens, enssol.u))
    # update ensemble
    update_ensemble!(ekp, enspred)
    # compute log joint probability (or likelihood if not EKS)
    logprobs = map((i,y) -> Inference.logprob(ekp, param_map, Θ[:,i], y), 1:N_ens, eachcol(enspred))
    return logprobs, enssol
end

function logprob(ekp::EnsembleKalmanProcess, pmap::ParameterMapping, θ, y)
    loglik = logpdf(MvNormal(ekp.obs_mean, ekp.obs_noise_cov), y)
    logdetJ⁻¹ = Inference.logprob(pmap, θ)
    return loglik + logdetJ⁻¹
end
function logprob(ekp::EnsembleKalmanProcess{FT,IT,<:Sampler}, pmap::ParameterMapping, θ, y) where {FT,IT}
    loglik = logpdf(MvNormal(ekp.obs_mean, ekp.obs_noise_cov), y)
    logprior = logpdf(MvNormal(ekp.process.prior_mean, ekp.process.prior_cov))
    logdetJ⁻¹ = Inference.logprob(pmap, θ)
    return loglik + logprior + logdetJ⁻¹
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
