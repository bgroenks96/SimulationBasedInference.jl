"""
    TuringPrior{varnames,TM<:Turing.Model} <: AbstractPrior

Represents a prior distribution formulated as a `Turing` model. The Turing model
can have any arbitrary structure, e.g. hierarchical or otherwise.
"""
struct TuringPrior{varnames,TM<:Turing.Model} <: AbstractPrior
    model::TM
    pnames::Vector{Symbol}
    function TuringPrior(model::Turing.Model)
        varnames = keys(Turing.VarInfo(model).metadata)
        sampled_param_names = extract_parameter_names(model)
        new{varnames,typeof(model)}(model, sampled_param_names)
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

# mandatory sampling dispatches
Base.rand(prior::TuringPrior) = rand(prior.model)
StatsBase.sample(rng::AbstractRNG, prior::TuringPrior, n::Int, args...; kwargs...) = sample(rng, prior.model, Prior(), n, args...; kwargs...)
