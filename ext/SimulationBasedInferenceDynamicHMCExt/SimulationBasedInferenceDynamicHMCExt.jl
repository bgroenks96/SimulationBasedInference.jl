module SimulationBasedInferenceDynamicHMCExt

using SimulationBasedInference

using LogDensityProblemsAD
using MCMCChains
using Statistics

import CommonSolve
import DynamicHMC
import Random

mutable struct DynamicHMCSolver{algType<:MCMC,probType,statsType,QType}
    sol::SimulatorInferenceSolution{algType,probType}
    steps::DynamicHMC.MCMCSteps
    stats::statsType
    Q::QType
end

function CommonSolve.init(
    prob::SimulatorInferenceProblem,
    mcmc::MCMC{<:DynamicHMC.NUTS};
    autodiff=:ForwardDiff,
    rng::Random.AbstractRNG=default_rng(),
)
    b = SBI.bijector(prob)
    q = b(sample(prob.prior))
    ℓ = ADgradient(autodiff, prob)
    # stepwise sampling; see DynamicHMC docs!
    # initialization
    results = DynamicHMC.mcmc_keep_warmup(rng, ℓ, 0; initialization=(; q), reporter = DynamicHMC.NoProgressReport())
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q = results.final_warmup_state.Q
    sol = SimulatorInferenceSolution(prob, mcmc, [], [], nothing)
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
    obs = map(retrieve, prob.forward_prob.observables)
    push!(sol.inputs, q)
    push!(sol.outputs, obs)
    push!(solver.stats, stats)
    return nothing
end

function CommonSolve.solve!(solver::DynamicHMCSolver)
    sol = solver.sol
    # iterate for N samples
    while length(sol.inputs) < sol.alg.nsamples
        step!(solver)
    end
    # construct transformed posterior chain
    prob = sol.prob
    b = SBI.bijector(prob)
    b⁻¹ = SBI.inverse(b)
    solver.sol.result = reduce(hcat, map(b⁻¹, sol.inputs))
    return solver.sol
end

end