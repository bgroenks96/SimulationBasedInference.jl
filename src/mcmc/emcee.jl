import AffineInvariantMCMC

Base.@kwdef struct Emcee end

function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:Emcee};
    storage::SimulationData=SimulatorArrayStorage(),
    nsamples = 1000,
    nchains = 100,
    thinning = 1,
    solve_kwargs...
)
    # log joint function;
    function loglik(θ)
        # deep copy inference problem to prevent memory
        # collisions between walkers; hopefully this doesn't
        # cause too much allocation...
        probcopy = deepcopy(prob)
        lp = logdensity(probcopy, θ)
        observables = probcopy.forward_prob.observables
        store!(storage, θ, observables)
        return lp
    end
    # sample prior and apply transform
    prior_samples = sample(prob.prior, mcmc.nchains)
    b = SBI.bijector(prob)
    θ₀ = b(prior_samples)
    samples, logprobs = AffineInvariantMCMC.sample(loglik, nchains, θ₀, nsamples, thinning)
    param_names = labels(prob.u0)
    chains = Chains(transpose(samples), param_names)
    return SimulatorInferenceSolution(prob, mcmc, storage, chains)
end
