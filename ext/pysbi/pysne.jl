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
    simulate_kwargs::NamedTuple
    sampling::samplingType
    data::Union{Missing,SimulationData}
    prior::Py
    simulator::Py
    inference::Py
    estimator::Union{Missing,Py}
    result::Union{Missing,Any}
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
        (; num_simulations, num_workers),
        sampling,
        missing,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
        missing
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
    # sample kwargs
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
        (;),
        sampling,
        data,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
        missing
    )
end

function CommonSolve.step!(solver::PySBISolver)
    # step 1: run simulations, if necessary
    if ismissing(solver.data)
        @info "Running simulations"
        Θ, Y = sbi.inference.simulate_for_sbi(solver.simulator; proposal=solver.prior, solver.simulate_kwargs...)
        solver.inference.append_simulations(Θ, Y)
        solver.data = SimulationArrayStorage()
        store!(solver.data, transpose(pyconvert(Matrix, Θ)), collect(eachrow(pyconvert(Matrix, Y))))
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

function CommonSolve.solve!(solver::PySBISolver)
    while step!(solver) end
    return SimulatorInferenceSolution(solver.prob, solver.alg, solver.data, solver.result)
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
