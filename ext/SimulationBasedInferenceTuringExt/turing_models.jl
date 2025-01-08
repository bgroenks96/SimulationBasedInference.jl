# Turing MCMC

SimulationBasedInference.MCMC(alg::Turing.InferenceAlgorithm, strat; kwargs...) = error("invalid sampling strategy $(typeof(strat)) for Turing algorithm")

function SimulationBasedInference.MCMC(
    alg::Turing.InferenceAlgorithm,
    strat::AbstractMCMC.AbstractMCMCEnsemble=MCMCSerial();
    kwargs...,
)
    return MCMC(alg, strat, (; kwargs...))
end

function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:Turing.InferenceAlgorithm};
    storage=SimulationArrayStorage(),
    num_samples=1000,
    num_chains=1,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...
)
    m = SBI.joint_model(prob, prob.forward_solver; storage, kwargs...)
    chain = Turing.sample(rng, m, mcmc.alg, mcmc.strat, num_samples, num_chains)
    return SimulatorInferenceSolution(prob, mcmc, storage, chain)
end

# model building functions.
function SimulationBasedInference.joint_model(
    inference_prob::SimulatorInferenceProblem{<:TuringSimulatorPrior},
    forward_alg;
    storage=SimulationArrayStorage(),
    solve_kwargs...
)
    # construct likelihood model
    lm = SBI.likelihood_model(inference_prob, forward_alg; storage, solve_kwargs...)
    # define Turing model
    @model function joint_model(::Type{T}=Float64) where {T}
        # sample parameters from prior model
        p_model = @submodel inference_prob.prior.model.model
        # sample parameters for each likelihood term
        p_lik = map(names(inference_prob)) do name
            lik = getproperty(inference_prob.likelihoods, name)
            p_i = @submodel prefix=$name likelihood_params(lik)
            name => p_i
        end
        p = ComponentVector(; model=p_model, p_lik...)
        # invoke submodel and retrieve observables + forward solution
        obs, sol = @submodel lm(p)
        return (; obs, sol, p)
    end
    m = joint_model()
    data = map(l -> l.data, inference_prob.likelihoods)
    return Turing.condition(m; data...)
end

function SimulationBasedInference.likelihood_model(
    inference_prob::SimulatorInferenceProblem{<:TuringSimulatorPrior},
    forward_alg;
    storage=SimulationArrayStorage(),
    solve_kwargs...
)
    @model function likelihood_model(ϕ::ComponentVector)
        # deepcopy problem to insure that concurrent/parallel evaluations of the model use fully independent memory
        inference_prob = deepcopy(inference_prob)
        forward_prob = inference_prob.forward_prob
        # collect observables form likelihood terms
        lik_observables = map(l -> l.obs, inference_prob.likelihoods)
        # recreate forward problem with merged observables;
        observables = merge(lik_observables, forward_prob.observables)
        forward_prob = SimulatorForwardProblem(forward_prob.prob, observables...)
        # solve forward problem
        forward_sol = solve(forward_prob, forward_alg; p=ϕ.model, solve_kwargs...)
        retcode = forward_sol.sol.retcode
        if retcode ∉ (ReturnCode.Default, ReturnCode.Success)
            # simulation failed, add -Inf logprob
            Turing.@addlogprob! -Inf
            return nothing, forward_sol
        end
        obs = map(names(inference_prob)) do name
            lik = getproperty(inference_prob.likelihoods, name)
            d = SBI.predictive_distribution(lik, ϕ[name])
            x ~ NamedDist(d, name)
            name => mean(d)
        end
        outputs = (; obs...)
        store!(storage, ϕ.model, outputs)
        return outputs, forward_sol
    end
end

@model function likelihood_params(lik::SimulatorLikelihood{<:Union{Normal,MvNormal}})
    σ ~ lik.prior.dist[1]
    return (; σ)
end
