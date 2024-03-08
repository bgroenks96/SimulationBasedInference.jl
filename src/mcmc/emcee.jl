import AffineInvariantMCMC

Base.@kwdef struct Emcee
    thinning::Int = 1
end

function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:Emcee};
    kwargs...
)
    # log joint function;
    # may need to deep copy `prob`?
    ℓ(θ) = logdensity(prob, θ)
    # sample prior and apply transform
    prior_samples = sample(prob.prior, mcmc.nchains)
    b = SBI.bijector(prob)
    θ₀ = b(prior_samples)
    samples, logprobs = AffineInvariantMCMC.sample(ℓ, mcmc.nchains, θ₀, mcmc.nsamples, mcmc.alg.thinning)
    param_names = labels(prob.u0)
    chains = Chains(transpose(samples), param_names)
    # TODO: how do we get the predictions...
    return SimulatorInferenceSolution(prob, mcmc, [], [], chains)
end
