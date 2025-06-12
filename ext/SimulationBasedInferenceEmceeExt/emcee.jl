
function CommonSolve.solve(
    prob::SimulatorInferenceProblem,
    emcee::typeof(AffineInvariantMCMC.sample);
    storage::SimulationData=SimulationArrayStorage(),
    num_samples = 1000,
    num_chains = 100,
    thinning = 1,
    rng::Random.AbstractRNG=Random.default_rng(),
    solve_kwargs...
)
    # make log density function;
    f = logdensityfunc(prob, storage)
    # sample prior and apply transform
    prior_samples = sample(rng, prob.prior, num_chains)
    ζ₀ = reduce(hcat, map(SBI.bijector(prob), prior_samples))
    samples, logprobs = emcee(f, num_chains, ζ₀, num_samples, thinning; rng)
    param_names = labels(prob.u0)
    chains = Chains(transpose(samples), param_names)
    return SimulatorInferenceSolution(prob, emcee, storage, chains)
end
