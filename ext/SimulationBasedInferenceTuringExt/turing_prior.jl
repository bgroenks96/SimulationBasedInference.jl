"""
    TuringPrior{varnames,TM<:Turing.Model,axesType} <: AbstractPrior

Represents a prior distribution formulated as a `Turing` model. The Turing model
can have any arbitrary structure, e.g. hierarchical or otherwise.
"""
struct TuringPrior{varnames,modelType<:Turing.Model,axesType} <: AbstractPrior
    model::modelType
    axes::axesType
    chain_names::Vector{Symbol}
    function TuringPrior(model::Turing.Model)
        varnames = keys(Turing.VarInfo(model).metadata)
        axes = getaxes(ComponentArray(rand(model)))
        chain_names = extract_parameter_names(model)
        new{varnames,typeof(model),typeof(axes)}(model, axes, chain_names)
    end
end

function (prior::TuringPrior)(θ::AbstractVector{T}) where {T}
    varinfo = Turing.DynamicPPL.VarInfo(prior.model);
    context = prior.model.context;
    new_vi = Turing.DynamicPPL.unflatten(varinfo, context, θ);
    pvec = first(Turing.DynamicPPL.evaluate!!(prior.model, new_vi, context))
    p = similar(pvec, T)
    copyto!(p, pvec)
    return p
end

SimulationBasedInference.prior(model::Turing.Model) = TuringPrior(model)

SimulationBasedInference.forward_map(prior::TuringPrior, θ::AbstractVector) = prior(θ)

SimulationBasedInference.logprob(prior::TuringPrior, θ::NamedTuple) = Turing.logprior(prior.model, θ)
function SimulationBasedInference.logprob(prior::TuringPrior, θ::AbstractVector)
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
Base.rand(rng::AbstractRNG, prior::TuringPrior) = ComponentArray(rand(rng, prior.model))

StatsBase.sample(rng::AbstractRNG, prior::TuringPrior, n::Int, args...; kwargs...) = [rand(rng, prior) for i in 1:n]

Bijectors.bijector(prior::TuringPrior) = bijector(prior.model)

function extract_parameter_names(m::Turing.Model)
    # sample chain to extract param names
    chain = sample(m, Prior(), 1, progress=false, verbose=false)
    return chain.name_map.parameters
end
