"""
    PySNE <: SBI.SimulatorInferenceAlgorithm

Wrapper for the sequential neural estimator (SNxE) algorithms from
the python `sbi` package. `algtype` should typically be one of `SNPE_A`, `SNPEB`,
`SNPE_C`, or 'SNLE_A', though any suitable algorithm type can be used.
"""
Base.@kwdef struct PySNE <: SBI.SimulatorInferenceAlgorithm
    algtype::Py = sbi.inference.SNPE_C
    density_estimator::String = "maf"
    device::String = "cpu"
    logging_level::String = "WARNING"
    summary_writer=nothing
    show_progress_bars=true
end

SNPE_A(kwargs...) = PySNE(; algtype=sbi.inference.SNPE_A, kwargs...)
SNPE_B(kwargs...) = PySNE(; algtype=sbi.inference.SNPE_B, kwargs...)
SNPE_C(kwargs...) = PySNE(; algtype=sbi.inference.SNPE_C, kwargs...)
SNLE_A(kwargs...) = PySNE(; algtype=sbi.inference.SNLE_A, kwargs...)

function build(alg::PySNE, prior)
    return alg.algtype(
        prior=prior,
        density_estimator=alg.density_estimator,
        device=alg.device,
        logging_level=alg.logging_level,
        summary_writer=alg.summary_writer,
        show_progress_bars=alg.show_progress_bars,
    )
end

"""
Base type for `sbi` sampling algorithms.
"""
abstract type PySBISampling end

"""
    PySBISolver{algType,samplingType<:PySBISampling}

Generic solver type for python sbi algorithms. Stores the prepared simulator,  prior,
simulation data, and inference algorithm.
"""
mutable struct PySBISolver{algType,samplingType<:PySBISampling}
    prob::SimulatorInferenceProblem
    alg::algType
    iter::Int
    maxiter::Int
    simulate_kwargs::NamedTuple
    train_kwargs::NamedTuple
    sampling::samplingType
    data::Union{Missing,SimulationData}
    proposal::Py
    simulator::Py
    inference::Py
    estimator::Union{Missing,Py}
end

function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::PySNE,
    simdata::Union{Nothing,SimulationData}=nothing,
    param_type::Type{T} = Vector;
    pred_transform = identity,
    prior = pyprior(inference_prob.prior),
    transform = SBI.unconstrained_forward_map(inference_prob.prior),
    rng::Random.AbstractRNG = Random.default_rng(),
    # simulate kwargs
    num_simulations::Int = 1000,
    num_workers::Int = 1,
    # training
    num_rounds::Int = 1,
    train_kwargs = (;),
    # sampling options
    sampling::PySBISampling = default_sampling(alg.algtype),
    # solve kwargs
    solve_kwargs...
) where {T}
    if isnothing(simdata)
        simdata = SimulationArrayStorage()
        pysim = pysimulator(inference_prob, simdata, transform, pred_transform, T; rng)
        prepared_prior, num_params, returns_numpy = sbi_utils.user_input_checks.process_prior(prior)
        prepared_sim = sbi_utils.user_input_checks.process_simulator(pysim, prepared_prior, returns_numpy)
        inference_alg = build(alg, prepared_prior)
        SBI.clear!(simdata)
    else
        pysim = pysimulator(inference_prob, simdata, transform, pred_transform, T; rng)
        prepared_prior, num_params, returns_numpy = sbi_utils.user_input_checks.process_prior(prior)
        prepared_sim = sbi_utils.user_input_checks.process_simulator(pysim, prepared_prior, returns_numpy)
        inference_alg = build(alg, prepared_prior)
        append_simulations!(inference_alg, inference_prob, simdata)
    end
    return PySBISolver(
        inference_prob,
        alg,
        1,
        num_rounds,
        (; num_simulations, num_workers),
        train_kwargs,
        sampling,
        simdata,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
    )
end

function CommonSolve.step!(solver::PySBISolver)
    if solver.iter > solver.maxiter
        return false
    end
    @info "Starting iteration $(solver.iter)/$(solver.maxiter)"
    # step 1: run simulations, if necessary
    if length(solver.data) == 0 || solver.iter > 1
        @info "Running simulations"
        Θ, Y = sbi.inference.simulate_for_sbi(solver.simulator; proposal=solver.proposal, solver.simulate_kwargs...)
        solver.inference.append_simulations(Θ, Y, solver.proposal)
    end

    # step 2: train density estimator
    @info "Training estimator"
    solver.estimator = solver.inference.train(; solver.train_kwargs...)

    # step 3: build posterior
    @info "Building posterior"
    x_obs = reduce(vcat, map(lik -> vec(lik.data), solver.prob.likelihoods))
    posterior = _build_posterior(solver.sampling, solver.inference, solver.estimator)
    posterior.set_default_x(Py(x_obs).to_numpy())
    solver.proposal = posterior
    solver.iter += 1
    return solver.iter <= solver.maxiter
end

function CommonSolve.solve!(solver::PySBISolver)
    while step!(solver) end
    posterior = SurrogatePosterior(solver.prob.prior, solver.proposal)
    return SimulatorInferenceSolution(solver.prob, solver.alg, solver.data, posterior)
end

# Sampling algorithms

struct DirectSampling <: PySBISampling
    parameters::Dict
    DirectSampling(; kwargs...) = new(Dict(map((k,v) -> string(k) => v, keys(kwargs), values(kwargs))))
end

struct RejectionSampling <: PySBISampling
    parameters::Dict
    RejectionSampling(; kwargs...) = new(Dict(map((k,v) -> string(k) => v, keys(kwargs), values(kwargs))))
end

struct MCMCSampling <: PySBISampling
    method::String
    parameters::Dict
    MCMCSampling(; method::String="slice_np", kwargs...) = new(method, Dict(map((k,v) -> string(k) => v, keys(kwargs), values(kwargs))))
end

function default_sampling(algtype)
    if pyconvert(Bool, algtype == sbi.inference.SNPE_A) ||
       pyconvert(Bool, algtype == sbi.inference.SNPE_B) ||
       pyconvert(Bool, algtype == sbi.inference.SNPE_C)
        DirectSampling()
    else
        MCMCSampling()
    end
end

function _build_posterior(sampling::DirectSampling, inference::Py, estimator::Py)
    direct_sampling_parameters = sampling.parameters
    return inference.build_posterior(estimator; sample_with="direct", direct_sampling_parameters)
end

function _build_posterior(sampling::RejectionSampling, inference::Py, estimator::Py)
    rejection_sampling_parameters = sampling.parameters
    return inference.build_posterior(estimator; sample_with="rejection", rejection_sampling_parameters)
end

function _build_posterior(sampling::MCMCSampling, inference::Py, estimator::Py)
    mcmc_method = sampling.method
    mcmc_parameters = sampling.parameters
    return inference.build_posterior(estimator; sample_with="mcmc", mcmc_method, mcmc_parameters)
end
