# Turing MCMC

SimulationBasedInference.MCMC(alg::Turing.InferenceAlgorithm, strat; kwargs...) = error("invalid sampling strategy $(typeof(strat)) for Turing algorithm")

function SimulationBasedInference.MCMC(
    alg::Turing.InferenceAlgorithm,
    strat::AbstractMCMC.AbstractMCMCEnsemble=MCMCSerial();
    kwargs...
)
    return MCMC(alg, strat; kwargs...)
end

function CommonSolve.solve(prob::SimulatorInferenceProblem, mcmc::MCMC{<:Turing.InferenceAlgorithm}; kwargs...)
    m = joint_model(prob, prob.forward_solver; kwargs...)
    m_cond = Turing.condition(m; prob.data...)
    chain = Turing.sample(m_cond, mcmc.alg, mcmc.strat, mcmc.nsamples, mcmc.nchains)
    return chain
end

# model building functions.
function SimulationBasedInference.joint_model(inference_prob::SimulatorInferenceProblem{<:TuringPrior}, forward_alg; solve_kwargs...)
    # construct likelihood model
    lm = likelihood_model(inference_prob, forward_alg; solve_kwargs...)
    # define Turing model
    @model function joint_model(::Type{T}=Float64) where {T}
        # sample parameters from prior model
        p_model = @submodel inference_prob.prior.model
        # sample parameters for each likelihood term
        p_lik = map(names(inference_prob)) do name
            lik = getproperty(prob.likelihoods, name)
            p_i = @submodel prefix=$name likelihood_params(lik)
            name => p_i
        end
        # collect parameters into named tuple
        p = (model=p_model, p_lik...)
        # invoke submodel and retrieve observables + forward solution
        obs, sol = @submodel lm(p)
        return (; obs, sol, p)
    end
    m = joint_model()
    return Turing.condition(m; inference_prob.data...)
end

function SimulationBasedInference.likelihood_model(inference_prob::SimulatorInferenceProblem{<:TuringPrior}, forward_alg; solve_kwargs...)
    @model function likelihood_model(p_model, p_lik::NamedTuple)
        # deepcopy problem to insure that concurrent/parallel evaluations of the model use fully independent memory
        inference_prob = deepcopy(inference_prob)
        forward_prob = inference_prob.forward_prob
        # collect observables form likelihood terms
        lik_observables = map(observable, inference_prob.likelihoods)
        # recreate forward problem with merged observables;
        observables = merge(lik_observables, forward_prob.observables)
        forward_prob = SimulatorForwardProblem(forward_prob.prob, observables)
        # solve forward problem
        forward_sol = solve(forward_prob, forward_alg; p=p_model, solve_kwargs...)
        retcode = forward_sol.sol.retcode
        if retcode ∉ (ReturnCode.Default, ReturnCode.Success)
            # simulation failed, add -Inf logprob
            Turing.@addlogprob! -Inf
            return nothing, forward_sol
        end
        obs = map(names(inference_prob)) do name
            lik = getproperty(inference_prob.likelihoods, name)
            d = predictive_distribution(lik, p_lik[name])
            x ~ NamedDist(d, name)
            name => mean(d)
        end
        return (; obs...), forward_sol
    end
end

@model function likelihood_params(lik::SimulatorLikelihood{<:Union{Normal,MvNormal}})
    σ ~ lik.prior
    return (; σ)
end
