"""
    GenSimulatorPrior{genType<:GenerativeFunction,argsType,axesType} <: AbstractSimulatorPrior

Represents a prior distribution formulated as a `Turing` model. The Turing model
can have any arbitrary structure, e.g. hierarchical or otherwise.
"""
struct GenSimulatorPrior{genType<:Gen.GenerativeFunction,argsType,choiceMapType,axesType} <: AbstractSimulatorPrior
    gf::genType
    args::argsType
    axes::axesType
    names::Vector{Symbol}
    choicemap_proto::choiceMapType # necessary for Gen.from_array
end

function GenSimulatorPrior(gf::Gen.GenerativeFunction, args::Tuple)
    trace = simulate(gf, args)
    choices = to_static_choice_map(choicemap(get_choices(trace)))
    choices_nt = choicemap2nt(choices)
    datatype = isempty(choices_nt) ? Float64 : eltype(choices_nt)
    params = ComponentVector{datatype}(choices_nt)
    names = Symbol.(labels(params))
    # use zero values for prototype choice map
    choices_proto = Gen.from_array(choices, getdata(zero(params)))
    return GenSimulatorPrior(gf, args, getaxes(params), names, choices_proto)
end

function (prior::GenSimulatorPrior)(θ::AbstractVector{T}) where {T}
    choices = Gen.from_array(prior.choicemap_proto, Vector(θ))
    trace, _ = generate(prior.gf, prior.args, choices)
    return Gen.get_retval(trace)
end

SimulationBasedInference.prior(model::Gen.GenerativeFunction, args::Tuple) = GenSimulatorPrior(model, args)

SimulationBasedInference.forward_map(prior::GenSimulatorPrior, θ::AbstractVector) = prior(θ)

SimulationBasedInference.logprob(prior::GenSimulatorPrior, θ::NamedTuple) = logprob(prior.model, ComponentVector(θ))
function SimulationBasedInference.logprob(prior::GenSimulatorPrior, θ::AbstractVector)
    choices = Gen.from_array(prior.choicemap_proto, Vector(θ))
    weight, retval = assess(prior.gf, prior.args, choices)
    return weight
end

# mandatory sampling dispatches
function Base.rand(rng::AbstractRNG, prior::GenSimulatorPrior)
    # TODO: use RNG once PR is merged
    # https://github.com/probcomp/Gen.jl/pull/520
    trace = simulate(prior.gf, prior.args)
    choices = to_static_choice_map(choicemap(get_choices(trace)))
    params = ComponentVector(choicemap2nt(choices))
    return params
end

Bijectors.bijector(prior::GenSimulatorPrior) = identity
