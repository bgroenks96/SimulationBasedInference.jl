const SNPE_A = sbi.inference.SNPE_A
const SNPE_B = sbi.inference.SNPE_B
const SNPE_C = sbi.inference.SNPE_C

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

function CommonSolve.step!(solver::PySBISolver{PySNPE})
    # step 1: run simulations, if necessary
    if ismissing(solver.data)
        @info "Running simulations"
        Θ, Y = sbi.inference.simulate_for_sbi(solver.simulator, proposal=solver.prior, num_simulations=solver.num_simulations)
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
    else
        @info "Building posterior"
        x_obs = reduce(vcat, map(lik -> vec(lik.data), solver.prob.likelihoods))
        solver.posterior = solver.inference.build_posterior(solver.estimator)
        solver.posterior.set_default_x(x_obs)
        return false
    end
end
