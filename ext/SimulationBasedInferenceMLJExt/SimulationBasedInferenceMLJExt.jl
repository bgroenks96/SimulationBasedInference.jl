module SimulationBasedInferenceMLJExt

using SimulationBasedInference
using SimulationBasedInference.Emulators

using MLJ

import MLJModelInterface as MMI

function Emulators.StackedEmulator(data::EmulatorData, model::MMI.Model; input_transform=NoTransform(), output_transform=NoTransform())
    X_tab = table(data.X)
    Yt = apply(transform, data.Y)
    machines = map(1:size(Yt,2)) do i
        yt_i = Yt[:,i]
        machine(model, X_tab, yt_i)
    end
    return Emulator(data, Emulators.stacked(machines); input_transform, output_transform)
end

function Emulators.fit!(em::Emulator{<:MLJ.Machine})
    em.model = MLJ.fit!(em.model)
    return em
end

function Emulators.fit!(em::Emulator{StackedRegressors{<:MLJ.Machine}}; mapfunc=map, kwargs...)
    fitted_models = mapfunc(em.model.models) do m
        MLJ.fit!(m)
    end
    em.model.models = fitted_models
    return em
end

Emulators.predict(m::MLJ.Machine, X) = MMI.predict(m, MMI.table(X))

end
