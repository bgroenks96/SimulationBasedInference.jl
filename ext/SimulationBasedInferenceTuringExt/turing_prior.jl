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
    p = first(Turing.DynamicPPL.evaluate!!(prior.model, new_vi, context))
    ϕ = similar(p, T)
    copyto!(ϕ, p)
    return ϕ
end

SimulationBasedInference.logdensity(prior::TuringPrior, θ::NamedTuple) = Turing.logprior(prior.model, θ)
function SimulationBasedInference.logdensity(prior::TuringPrior, θ::AbstractVector)
    return Turing.logprior(prior.model, ComponentVector(getdata(θ), prior.axes))
end

# mandatory sampling dispatches
Base.rand(rng::AbstractRNG, prior::TuringPrior) = ComponentArray(rand(rng, prior.model))

StatsBase.sample(rng::AbstractRNG, prior::TuringPrior, n::Int, args...; kwargs...) = [rand(rng, prior) for i in 1:n]

Bijectors.bijector(prior::TuringPrior) = bijector(prior.model)
