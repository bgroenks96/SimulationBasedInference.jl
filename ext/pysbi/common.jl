function pysimulator(inference_prob::SimulatorInferenceProblem, transform, pred_transform, ::Type{T}=Vector; rng::Random.AbstractRNG=Random.default_rng()) where {T}
    function simulator(ζ::AbstractVector, return_py::Bool=true)
        θ = zero(inference_prob.u0) + ζ
        ϕ = SBI.forward_map(inference_prob.prior, θ)
        solve(inference_prob.forward_prob, inference_prob.forward_solver, p=ϕ.model)
        obs_vecs = map(inference_prob.likelihoods) do lik
            dist = SBI.predictive_distribution(lik, ϕ[nameof(lik)])
            pred_transform(rand(rng, dist))
            # pred_transform(mean(dist))
        end
        if return_py
            return Py(reduce(vcat, obs_vecs)).to_numpy()
        else
            return reduce(vcat, obs_vecs)
        end
    end
    simulator(ζ::PyIterable, return_py::Bool=true) = simulator(pyconvert(Vector, ζ), return_py)
    simulator(ζ::PyMatrix) = Py(transpose(reduce(hcat, map(x -> simulator(x, false), eachrow(ζ))))).to_numpy()
    return simulator
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

"""
    SurrogatePosterior

Wraps the posterior object returned by `sbi` along with the original prior.
"""
struct SurrogatePosterior
    prior::AbstractPrior
    posterior::Py
end

function SBI.logprob(sp::SurrogatePosterior, x::AbstractVector; transform=bijector(sp.prior), kwargs...)
    return pyconvert(Vector, sp.posterior.log_prob(transform(x); kwargs...))[1] + logabsdetjac(transform, x)
end

StatsBase.sample(sp::SurrogatePosterior, n::Int; kwargs...) = sample(Random.default_rng(), sp, n; kwargs...)
function StatsBase.sample(rng::Random.AbstractRNG, sp::SurrogatePosterior, n::Int; obs=nothing, transform=inverse(bijector(sp.prior)))
    # TODO: check why posterior.sample doesn't take random seed...?
    # seed = isa(rng, Random.TaskLocalRNG) ? Random.GLOBAL_SEED : seed.rng
    if isnothing(obs)
        raw_samples = pyconvert(Matrix, sp.posterior.sample((n,)))
    else
        x = Py(obs).to_numpy()
        raw_samples = pyconvert(Matrix, sp.posterior.sample((n,), x=x))
    end
    return reduce(hcat, map(transform, eachrow(raw_samples)))
end

abstract type PySBISampling end

Base.@kwdef struct DirectSampling <: PySBISampling
    max_sampling_batch_size::Int = 10_000
    device::Union{Nothing,String} = nothing
    x_shape::Union{Nothing,Dims} = nothing
    enable_transform::Bool = true
end

Base.@kwdef struct MCMCSampling <: PySBISampling
    method::String = "slice_np"
    thin::Int = -1
    warmup_steps::Int = 200
    num_chains::Int = 1
    init_strategy::String = "resample"
    init_strategy_parameters::Union{Nothing,Dict} = nothing
    init_strategy_num_candidates::Union{Nothing,Int} = nothing
    num_workers::Int = 1
    mp_context::String = "spawn"
    device::Union{Nothing,String} = nothing
end
