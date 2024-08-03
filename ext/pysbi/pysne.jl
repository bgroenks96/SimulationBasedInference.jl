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
    param_type::Type{T} = Vector;
    transform = inverse(bijector(inference_prob)),
    pred_transform = identity,
    prior = pyprior(inference_prob.prior),
    rng::Random.AbstractRNG = Random.default_rng(),
    # simulate kwargs
    num_simulations::Int = 1000,
    num_workers::Int = 1,
    # training
    num_rounds::Int = 1,
    train_kwargs = (;),
    # sample kwargs
    sampling::PySBISampling = default_sampling(alg.algtype),
    # solve kwargs
    solve_kwargs...
) where {T}
    pysim = pysimulator(inference_prob, transform, pred_transform, T; rng)
    prepared_sim, prepared_prior = sbi.inference.prepare_for_sbi(pysim, prior)
    inference_alg = build(alg, prepared_prior)
    return PySBISolver(
        inference_prob,
        alg,
        1,
        num_rounds,
        (; num_simulations, num_workers),
        train_kwargs,
        sampling,
        missing,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
    )
end

function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::PySNE,
    data::SimulationData,
    param_type::Type{T} = Vector;
    transform = inverse(bijector(inference_prob)),
    pred_transform = identity,
    prior = pyprior(inference_prob.prior),
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
    pysim = pysimulator(inference_prob, transform, pred_transform, T; rng)
    prepared_sim, prepared_prior = sbi.inference.prepare_for_sbi(pysim, prior)
    inference_alg = build(alg, prepared_prior)
    append_simulations!(inference_alg, inference_prob, data)
    return PySBISolver(
        inference_prob,
        alg,
        1,
        num_rounds,
        (; num_simulations, num_workers),
        train_kwargs,
        sampling,
        data,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
    )
end

function CommonSolve.step!(solver::PySBISolver)
    @info "Starting iteration $(solver.iter)/$(solver.maxiter)"
    # step 1: run simulations, if necessary
    if ismissing(solver.data) || solver.iter > 1
        @info "Running simulations"
        Θ, Y = sbi.inference.simulate_for_sbi(solver.simulator; proposal=solver.proposal, solver.simulate_kwargs...)
        solver.inference.append_simulations(Θ, Y, solver.proposal)
        # Create internal simulation storage if it does not exist yet;
        # Note that this is duplicating simulation data already stored by the python package which is less than ideal.
        # It would be good to consider how to eliminate this redundancy in the future.
        solver.data = ismissing(solver.data) ? SimulationArrayStorage() : solver.data
        store!(solver.data, transpose(pyconvert(Matrix, Θ)), collect(eachrow(pyconvert(Matrix, Y))))
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

abstract type PySBISampling end

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
        RejectionSampling() # TODO: update with next sbi release
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
