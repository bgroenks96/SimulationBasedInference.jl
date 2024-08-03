function pysimulator(inference_prob::SimulatorInferenceProblem, data::SimulationData, pred_transform, ::Type{T}=Vector; rng::Random.AbstractRNG=Random.default_rng()) where {T}
    function simulator(ζ::AbstractVector, return_py::Bool=true)
        θ = zero(inference_prob.u0) + ζ
        ϕ = SBI.forward_map(inference_prob.prior, θ)
        forward_sol = solve(inference_prob.forward_prob, inference_prob.forward_solver, p=ϕ.model)
        store!(data, θ, map(getvalue, forward_sol.prob.observables))
        obs_vecs = map(inference_prob.likelihoods) do lik
            if hasproperty(ϕ, nameof(lik))
                y_pred = SBI.sample_prediction(lik, ϕ[nameof(lik)])
            else
                y_pred = SBI.sample_prediction(lik)
            end
            pred_transform(y_pred)
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
    prior::AbstractSimulatorPrior
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
