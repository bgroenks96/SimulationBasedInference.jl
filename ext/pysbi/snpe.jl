const SNPE_A = sbi.inference.SNPE_A
const SNPE_B = sbi.inference.SNPE_B
const SNPE_C = sbi.inference.SNPE_C

"""
    PySNPE <: PySBIAlgorithm

Wrapper for the sequential neural posterior estimator (SNPE) algorithms from
the python `sbi` package. `algtype` should typically be one of `SNPE_A`, `SNPEB`,
or `SNPE_C`, though any suitable SNPE algorithm type can be used.
"""
Base.@kwdef struct PySNPE <: PySBIAlgorithm
    algtype::Py = SNPE_C
    density_estimator::String = "maf"
    device::String = "cpu"
    logging_level::String = "WARNING"
    summary_writer=nothing
    show_progress_bars=true
end

function (alg::PySNPE)()
    return alg.algtype(
        density_estimator=alg.density_estimator,
        device=alg.device,
        logging_level=alg.logging_level,
        summary_writer=alg.summary_writer,
        show_progress_bars=alg.show_progress_bars,
    )
end

struct SurrogatePosterior
    prior::AbstractPrior
    posterior::Py
end

function SBI.logprob(sp::SurrogatePosterior, x::AbstractVector; transform=bijector(sp.prior), kwargs...)
    return pyconvert(Vector, sp.posterior.log_prob(transform(x); kwargs...))[1] + logabsdetjac(transform, x)
end

StatsBase.sample(sp::SurrogatePosterior, n::Int; kwargs...) = sample(Random.default_rng(), sp, n; kwargs...)
function StatsBase.sample(rng::Random.AbstractRNG, sp::SurrogatePosterior, n::Int; transform=inverse(bijector(sp.prior)))
    # TODO: check why posterior.sample doesn't take random seed...?
    # seed = isa(rng, Random.TaskLocalRNG) ? Random.GLOBAL_SEED : seed.rng
    raw_samples = pyconvert(Matrix, sp.posterior.sample((n,)))
    return reduce(hcat, map(transform, eachrow(raw_samples)))
end

function CommonSolve.step!(solver::PySBISolver{PySNPE})
    # step 1: run simulations, if necessary
    if ismissing(solver.data)
        @info "Running simulations"
        Θ, Y = sbi.inference.simulate_for_sbi(solver.simulator; proposal=solver.prior, solver.simulate_kwargs...)
        solver.inference.append_simulations(Θ, Y)
        solver.data = SimulationArrayStorage()
        store!(solver.data, pyconvert(Matrix, Θ), collect(eachrow(pyconvert(Matrix, Y))))
        return true
    # step 2: train density estimator
    elseif ismissing(solver.estimator)
        @info "Training estimator"
        solver.estimator = solver.inference.train()
        return true
    # step 3: build posterior
    elseif ismissing(solver.result)
        @info "Building posterior"
        x_obs = reduce(vcat, map(lik -> vec(lik.data), solver.prob.likelihoods))
        posterior = solver.inference.build_posterior(solver.estimator)
        posterior.set_default_x(x_obs)
        solver.result = SurrogatePosterior(solver.prob.prior, posterior)
        return false
    else
        return false
    end
end
