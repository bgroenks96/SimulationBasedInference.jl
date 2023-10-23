Base.@kwdef struct TuringMCMC{Talg,Tstrat} <: SimulatorInferenceAlgorithm
    alg::Talg = MH()
    strat::Tstrat = MCMCSerial()
    num_samples::Int = 1000
    num_chains::Int = 2
end

# model building functions.

function joint_model(inference_prob::SimulatorInferenceProblem{<:TuringPrior}, ode_alg; solve_kwargs...)
    @model function joint_model(::Type{T}=Float64) where {T}
        p = @submodel inference_prob.prior.model
        obs, sol = @submodel likelihood_model(inference_prob, ode_alg, p; solve_kwargs...)
        return (; obs, sol, p)
    end
    m = joint_model()
    return Turing.condition(m; inference_prob.data...)
end

@model function likelihood_model(inference_prob::SimulatorInferenceProblem{<:TuringPrior}, ode_alg, p; solve_kwargs...)
    # deepcopy problem to insure that concurrent/parallel evaluations of the model use fully independent memory
    inference_prob = deepcopy(inference_prob)
    forward_prob = inference_prob.forward_prob
    # collect observables form likelihood terms
    lik_observables = map(observable, inference_prob.likelihoods)
    # recreate forward problem with merged observables;
    observables = merge(lik_observables, forward_prob.observables)
    forward_prob = SimulatorForwardProblem(forward_prob.prob, observables)
    # solve forward problem
    forward_sol = solve(forward_prob, ode_alg; p=p, solve_kwargs...)
    retcode = forward_sol.sol.retcode
    if retcode ∉ (ReturnCode.Default, ReturnCode.Success)
        # simulation failed, add -Inf logprob
        Turing.@addlogprob! -Inf
        return nothing, forward_sol
    end
    obs = map(names(inference_prob)) do name
        lik = getproperty(inference_prob.likelihoods, name)
        lik_params = @submodel prefix=$name likelihood_params(lik)
        d = lik(lik_params...)
        x ~ NamedDist(d, name)
        name => mean(d)
    end
    return (; obs...), forward_sol
end

@model function likelihood_params(lik::MvGaussianLikelihood)
    σ ~ lik.prior
    return (; σ)
end

function CommonSolve.solve(prob::SimulatorInferenceProblem, mcmc::TuringMCMC, ode_alg; kwargs...)
    m = joint_model(prob, ode_alg; kwargs...)
    m_cond = Turing.condition(m; prob.data...)
    chain = Turing.sample(m_cond, mcmc.alg, mcmc.strat, mcmc.num_samples, mcmc.num_chains)
    return chain
end
