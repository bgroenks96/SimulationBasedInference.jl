"""
    GenSimulatorPrior{genType<:GenerativeFunction,argsType,selType,axesType,paraTypes,staticTypes} <: AbstractSimulatorPrior

Represents a prior distribution formulated as a `Gen` stochastic function.
"""
struct GenSimulatorPrior{genType<:Gen.GenerativeFunction,argsType,selType,axesType,paraTypes,staticTypes} <: AbstractSimulatorPrior
    gf::genType
    args::argsType
    axes::axesType
    names::Vector{Symbol}
    sel::selType
    param_choices::paraTypes
    static_choices::staticTypes
end

function GenSimulatorPrior(
    gf::Gen.GenerativeFunction,
    args::Tuple,
    sel::Selection=selectall(),
    preset_choices::ChoiceMap=choicemap(),
)
    trace, _ = generate(gf, args, preset_choices)
    choices = to_static_choice_map(get_selected(choicemap(get_choices(trace)), sel))
    choices_nt = choicemap2nt(choices)
    datatype = isempty(choices_nt) ? Float64 : eltype(choices_nt)
    params = ComponentVector{datatype}(choices_nt)
    names = Symbol.(labels(params))
    # use zero values for prototype choice map
    para_choices = Gen.from_array(choices, getdata(zero(params)))
    static_choices = to_static_choice_map(preset_choices)
    return GenSimulatorPrior(gf, args, getaxes(params), names, sel, para_choices, static_choices)
end

function (prior::GenSimulatorPrior)(θ::AbstractVector{T}) where {T}
    # create choice map from parameter array
    choices = Gen.from_array(prior.param_choices, Vector(θ))
    # merge with conditioned static choices
    trace, _ = generate(prior.gf, prior.args, merge(prior.static_choices, choices))
    return Gen.get_retval(trace)
end

SimulationBasedInference.prior(
    model::Gen.GenerativeFunction,
    args::Tuple,
    sel::Selection=Gen.selectall(),
    choices::ChoiceMap=choicemap(),
) = GenSimulatorPrior(model, args, sel, choices)

SimulationBasedInference.forward_map(prior::GenSimulatorPrior, θ::AbstractVector) = prior(θ)

SimulationBasedInference.logprob(prior::GenSimulatorPrior, θ::NamedTuple) = logprob(prior.model, ComponentVector(θ))
function SimulationBasedInference.logprob(prior::GenSimulatorPrior, θ::AbstractVector)
    choices = Gen.from_array(prior.param_choices, Vector(θ))
    weight, retval = assess(prior.gf, prior.args, choices)
    return weight
end

# mandatory sampling dispatches
function Base.rand(rng::AbstractRNG, prior::GenSimulatorPrior)
    # TODO: use RNG once PR is merged
    # https://github.com/probcomp/Gen.jl/pull/520
    trace, _ = generate(prior.gf, prior.args, prior.static_choices)
    choices = to_static_choice_map(get_selected(choicemap(get_choices(trace)), prior.sel))
    params = ComponentVector(choicemap2nt(choices))
    return params
end

Bijectors.bijector(prior::GenSimulatorPrior) = identity
