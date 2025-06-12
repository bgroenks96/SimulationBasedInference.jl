"""
    TuringSimulatorPrior{TM<:Model,axesType} <: AbstractSimulatorPrior

Represents a prior distribution formulated as a `Turing` model. The Turing model
can have any arbitrary structure, e.g. hierarchical or otherwise.
"""
struct TuringSimulatorPrior{modelType<:Model,axesType} <: AbstractSimulatorPrior
    model::modelType
    axes::axesType
    chain_names::Vector{Symbol}
    function TuringSimulatorPrior(model::Model)
        axes = getaxes(ComponentArray(rand(model)))
        chain_names = extract_parameter_names(model)
        new{typeof(model),typeof(axes)}(model, axes, chain_names)
    end
end

function (prior::TuringSimulatorPrior)(θ::AbstractVector{T}) where {T}
    varinfo = Turing.DynamicPPL.VarInfo(prior.model);
    context = prior.model.context;
    new_vi = Turing.DynamicPPL.unflatten(varinfo, context, θ);
    pvec = first(Turing.DynamicPPL.evaluate!!(prior.model, new_vi, context))
    p = similar(pvec, T)
    copyto!(p, pvec)
    return p
end

SimulationBasedInference.prior(model::Model) = TuringSimulatorPrior(model)

SimulationBasedInference.forward_map(prior::TuringSimulatorPrior, θ::AbstractVector) = prior(θ)

SimulationBasedInference.logprob(prior::TuringSimulatorPrior, θ::NamedTuple) = Turing.logprior(prior.model, θ)
function SimulationBasedInference.logprob(prior::TuringSimulatorPrior, θ::AbstractVector)
    ϕ = ComponentVector(getdata(θ), prior.axes)
    # here we have to do some nasty Turing manipulation to make sure this function is
    # autodiff compatible; basically we have to reconstruct `varinfo` based on the type of ϕ.
    varinfo = Turing.DynamicPPL.VarInfo(prior.model);
    context = prior.model.context;
    # constructs a new VarInfo type from the parameter values
    new_vi = Turing.DynamicPPL.unflatten(varinfo, context, ϕ);
    return Turing.logprior(prior.model, new_vi)
end

# mandatory sampling dispatches
Base.rand(rng::AbstractRNG, prior::TuringSimulatorPrior) = ComponentArray(rand(rng, prior.model))

StatsBase.sample(rng::AbstractRNG, prior::TuringSimulatorPrior, n::Int, args...; kwargs...) = [rand(rng, prior) for i in 1:n]

Bijectors.bijector(prior::TuringSimulatorPrior) = bijector(prior.model)

function extract_parameter_names(m::Model)
    # sample chain to extract param names
    chain = sample(m, Prior(), 1, progress=false, verbose=false)
    return chain.name_map.parameters
end
