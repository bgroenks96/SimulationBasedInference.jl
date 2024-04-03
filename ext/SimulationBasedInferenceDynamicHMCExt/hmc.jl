function CommonSolve.init(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:DynamicHMC.NUTS};
    autodiff=:ForwardDiff,
    rng::Random.AbstractRNG=Random.default_rng(),
    storage::SBI.SimulationData=SimulationArrayStorage(),
    warmup_reporter=DynamicHMC.NoProgressReport(),
)
    b = SBI.bijector(prob)
    q = b(sample(prob.prior))
    ℓ = ADgradient(autodiff, prob)
    # stepwise sampling; see DynamicHMC docs!
    # initialization
    results = DynamicHMC.mcmc_keep_warmup(rng, ℓ, 0; initialization=(; q), reporter = warmup_reporter)
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q = results.final_warmup_state.Q
    sol = SimulatorInferenceSolution(prob, mcmc, storage, nothing)
    return DynamicHMCSolver(sol, steps, results.inference.tree_statistics, Q)
end

function CommonSolve.step!(solver::DynamicHMCSolver)
    sol = solver.sol
    prob = sol.prob
    # a single update step
    solver.Q, stats = DynamicHMC.mcmc_next_step(solver.steps, solver.Q)
    # extract the position
    q = solver.Q.q
    # extract observables
    obs = map(obs -> ForwardDiff.value.(retrieve(obs)), prob.forward_prob.observables)
    store!(sol.storage, q, obs)
    push!(solver.stats, stats)
    return nothing
end

function CommonSolve.solve!(solver::DynamicHMCSolver)
    sol = solver.sol
    # iterate for N samples
    while length(sol.storage) < sol.alg.nsamples
        step!(solver)
    end
    # construct transformed posterior chain
    prob = sol.prob
    b = SBI.bijector(prob)
    b⁻¹ = SBI.inverse(b)
    samples = transpose(reduce(hcat, map(b⁻¹, getinputs(sol.storage))))
    param_names = labels(prob.u0)
    solver.sol.result = Chains(reshape(samples, size(samples)..., 1), param_names)
    return solver.sol
end
