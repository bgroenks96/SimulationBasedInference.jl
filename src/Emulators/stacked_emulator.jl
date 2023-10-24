mutable struct StackedMLEmulator{TM} <: Emulator
    data::EmulatorData
    transform::EmulatorDataTransform
    models::TM
end

function StackedMLEmulator(model::MMI.Model, data::EmulatorData, transform=Decorrelated(data))
    X_tab = MMI.table(transpose(data.X))
    Yt = transform_target(transform, data.Y)
    rank = length(transform.truncated_svd.S)
    machines = map(1:rank) do i
        yt_i = Yt[i,:]
        machine(model, X_tab, yt_i)
    end
    return StackedMLEmulator(data, transform, machines)
end

function MMI.predict(em::StackedMLEmulator, X_new::AbstractMatrix; obs_noise=1e-6)
    preds = map(em.models) do m
        MMI.predict(m, MMI.table(X_new))
    end
    dists = reduce(hcat, preds)
    μ_pred = mean.(dists)
    σ²_pred = var.(dists)
    V = em.transform.truncated_svd.V
    sqrt_S = Diagonal(sqrt.(em.transform.truncated_svd.S))
    μs = map(eachrow(μ_pred)) do μ
        inverse_transform_target(em.transform, reshape(μ, :, 1))[:,1]
    end
    Σs = map(eachrow(σ²_pred)) do σ²
        Symmetric(V*sqrt_S*Diagonal(σ²)*sqrt_S*V') + obs_noise*I
    end
    return map(MvNormal, μs, Σs)
end
