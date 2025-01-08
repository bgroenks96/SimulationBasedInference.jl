mutable struct DynamicHMCSolver{algType<:MCMC,probType,statsType,QType}
    sol::SimulatorInferenceSolution{algType,probType}
    num_samples::Int
    num_chains::Int
    steps::DynamicHMC.MCMCSteps
    stats::statsType
    Q::QType
end

function default_hmc_init(rng, prob::SimulatorInferenceProblem)
    b = SBI.bijector(prob)
    q = b(sample(rng, prob.prior))
    return (; q)
end

function CommonSolve.init(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:DynamicHMC.NUTS};
    num_samples=1000,
    num_chains=1,
    autodiff=:ForwardDiff,
    rng::Random.AbstractRNG=Random.default_rng(),
    storage::SBI.SimulationData=SimulationArrayStorage(),
    initialization=default_hmc_init(rng, prob),
    warmup_stages=DynamicHMC.default_warmup_stages(),
    warmup_reporter=DynamicHMC.NoProgressReport(),
    solve_kwargs...,
)
    b = SBI.bijector(prob)
    q = b(sample(rng, prob.prior))
    ℓ = ADgradient(autodiff, logdensity(prob; solve_kwargs...))
    # stepwise sampling; see DynamicHMC docs!
    # initialization
    results = DynamicHMC.mcmc_keep_warmup(rng, ℓ, 0; initialization, warmup_stages, reporter = warmup_reporter)
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q = results.final_warmup_state.Q
    sol = SimulatorInferenceSolution(prob, mcmc, storage, nothing)
    return DynamicHMCSolver(sol, num_samples, num_chains, steps, results.inference.tree_statistics, Q)
end

function CommonSolve.step!(solver::DynamicHMCSolver)
    sol = solver.sol
    prob = sol.prob
    # a single update step
    solver.Q, stats = DynamicHMC.mcmc_next_step(solver.steps, solver.Q)
    # extract the position
    q = solver.Q.q
    # extract observables
    obs = map(obs -> ForwardDiff.value.(getvalue(obs)), prob.forward_prob.observables)
    store!(sol.storage, q, obs)
    push!(solver.stats, stats)
    return nothing
end

function CommonSolve.solve!(solver::DynamicHMCSolver)
    sol = solver.sol
    # iterate for N samples
    while length(sol.storage) < solver.num_samples
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
