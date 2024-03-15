module SimulationBasedInferenceFluxExt

using SimulationBasedInference
using SimulationBasedInference.Emulators

using Flux

mutable struct FluxModel{modelType,lossType,optType,paramsType,reType}
    model::modelType
    loss::lossType
    opt::optType
    params::paramsType
    re::reType
end

function Emulators.Emulator(
    data::EmulatorData,
    model::Flux.Chain;
    optimizer=Flux.AMSGrad(),
    loss=Flux.Losses.mse, kwargs...
)
    p, re = Flux.destructure(model)
    data_converted = EmulatorData(convert.(eltype(p), data.X), convert.(eltype(p), data.Y))
    return Emulator(data_converted, FluxModel(model, loss, optimizer, p, re); kwargs...)
end

function Emulators.fit!(m::FluxModel, X, Y; epochs=10, verbose=false, kwargs...)
    # set up optimizer
    state = Flux.setup(m.opt, m.model)
    # construct data pairs
    data = map(tuple, eachcol(X), eachcol(Y))
    # train model
    for j in 1:epochs
        ∑ℓ = 0.0
        for (xᵢ, yᵢ) in data
            ℓ, ∂ℓ∂p = Flux.withgradient(m.model, xᵢ, yᵢ) do f, xᵢ, yᵢ
                predᵢ = f(xᵢ)
                m.loss(predᵢ, yᵢ)
            end
            Flux.update!(state, m.model, ∂ℓ∂p[1])
            ∑ℓ += ℓ
        end
        if verbose
            @info "Finished epoch $j, training loss: $∑ℓ"
        end
    end
    return m
end

function Emulators.predict(m::FluxModel, X)
    return m.model(X)
end

end