"""
    Emulator{TM,TT} <: Emulator

Data structure consisting of some training data for a model emulator, appropriate transforms,
and a tuple of univariate regressors which are applied to the transformed data.
"""
mutable struct Emulator{modelType,xtType<:DataTransform,ytType<:DataTransform}
    data::EmulatorData
    model::modelType
    input_transform::xtType
    output_transform::ytType
end

Emulator(data::EmulatorData, model; input_transform=NoTransform(), output_transform=NoTransform()) = Emulator(data, model, input_transform, output_transform)

function StackedEmulator(data::EmulatorData, model; input_transform=NoTransform(), output_transform=NoTransform())
    Yt = apply(output_transform, data.Y)
    return Emulator(data, stacked(model, size(Yt, 2)), input_transform, output_transform)
end

function fit!(em::Emulator; kwargs...)
    data = em.data
    Xt = apply(em.input_transform, data.X)
    Yt = apply(em.output_transform, data.Y)
    em.model = fit!(em.model, Xt, Yt, data.static_inputs...; kwargs...)
    return em
end

function predict(em::Emulator, X::AbstractMatrix)
    Xt = apply(em.input_transform, X)
    Yp = predict(em.model, Xt, map(s -> repeat(s, 1, size(Xt, 2)), em.data.static_inputs)...)
    return apply_inverse(em.output_transform, Yp)
end
predict(em::Emulator, x::AbstractVector) = predict(em, reshape(x, :, 1))
