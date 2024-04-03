module PySBI

using ..PythonCall

using Bijectors
using CommonSolve
using SimulationBasedInference

import Random

const torch = pyimport("torch")

# import sbi and its submodules
const sbi = pyimport("sbi")
pyimport("sbi.inference")
pyimport("sbi.utils")

abstract type PySBIAlgorithm end

"""
    PySBISolver

Generic solver type for python sbi algorithms. Stores the prepared simulator,  prior,
simulation data, and inference algorithm.
"""
mutable struct PySBISolver{algType<:PySBIAlgorithm}
    prob::SimulatorInferenceProblem
    alg::algType
    simulate_kwargs::NamedTuple
    data::Union{Missing,SimulationData}
    prior::Py
    simulator::Py
    inference::Py
    estimator::Union{Missing,Py}
    result::Union{Missing,Any}
end

function pysimulator(inference_prob::SimulatorInferenceProblem, transform, ::Type{T}=Vector; rng::Random.AbstractRNG=Random.default_rng()) where {T}
    function simulator(ζ)
        θ = zero(inference_prob.u0) + transform(pyconvert(T, ζ))
        ϕ = SBI.forward_map(inference_prob.prior, θ)
        solve(inference_prob.forward_prob, inference_prob.forward_solver, p=ϕ.model)
        obs_vecs = map(inference_prob.likelihoods) do lik
            dist = SBI.predictive_distribution(lik, ϕ[nameof(lik)])
            rand(rng, dist)
        end
        return Py(reduce(vcat, obs_vecs)).to_numpy()
    end
end

function append_simulations!(inference_alg::Py, inference_prob::SimulatorInferenceProblem, data::SimulationData)
    inputs = getinputs(data)
    outputs = map(getoutputs(data)) do out
        reduce(vcat, map(k -> vec(out[k]), keys(inference_prob.likelihoods)))
    end
    Θ = transpose(reduce(hcat, inputs))
    Y = transpose(reduce(hcat, outputs))
    inference_alg.append_simulations(torch.Tensor(Py(Θ).to_numpy()), torch.Tensor(Py(Y).to_numpy()))
end

function CommonSolve.init(
    inference_prob::SimulatorInferenceProblem,
    alg::PySBIAlgorithm,
    param_type::Type{T} = Vector;
    transform = inverse(bijector(inference_prob)),
    pyprior = torchprior(inference_prob.prior),
    rng::Random.AbstractRNG = Random.default_rng(),
    num_simulations::Int = 1000,
    num_workers::Int = 1,
    solve_kwargs...
) where {T}
    pysim = pysimulator(inference_prob, transform, T; rng)
    prepared_sim, prepared_prior = sbi.inference.prepare_for_sbi(pysim, pyprior)
    inference_alg = alg()
    return PySBISolver(
        inference_prob,
        alg,
        (; num_simulations, num_workers),
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
    alg::PySBIAlgorithm,
    data::SimulationData,
    param_type::Type{T} = Vector;
    transform = inverse(bijector(inference_prob)),
    pyprior = torchprior(inference_prob.prior),
    rng::Random.AbstractRNG = Random.default_rng(),
    solve_kwargs...
) where {T}
    pysim = pysimulator(inference_prob, transform, T; rng)
    prepared_sim, prepared_prior = sbi.inference.prepare_for_sbi(pysim, pyprior)
    inference_alg = alg()
    append_simulations!(inference_alg, inference_prob, data)
    return PySBISolver(
        inference_prob,
        alg,
        (;),
        data,
        prepared_prior,
        prepared_sim,
        inference_alg,
        missing,
        missing
    )
end

function CommonSolve.solve!(solver::PySBISolver)
    while step!(solver) end
    return SimulatorInferenceSolution(solver.prob, solver.alg, solver.data, solver.result)
end

export torchprior
include("torchpriors.jl")

export PySNPE
include("snpe.jl")

export PySNLE
include("snle.jl")

end
