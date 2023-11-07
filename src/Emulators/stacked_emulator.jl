"""
    StackedMLEmulator{TM} <: Emulator

Data structure consisting of some training data for a model emulator, a tranform operation,
and a tuple of univariate regressors which are applied to the transformed data.
"""
mutable struct StackedMLEmulator{TM} <: Emulator
    data::EmulatorData
    transform::EmulatorDataTransform
    models::TM
end

function StackedMLEmulator(model::MMI.Model, data::EmulatorData, transform=NoTransform())
    X_tab = MMI.table(data.X)
    Yt = transform_target(transform, data.Y)
    rank = length(transform.truncated_svd.S)
    machines = map(1:rank) do i
        yt_i = Yt[:,i]
        machine(model, X_tab, yt_i)
    end
    return StackedMLEmulator(data, transform, machines)
end

MLJBase.fit!(em::StackedMLEmulator; mapper=map) = StackedMLEmulator(em.data, em.transform, mapper(MLJBase.fit!, em.models))

MMI.predict(em::StackedMLEmulator, X_new::AbstractMatrix) = _predict(eltype(em.models), em, X_new)

function _predict(::Type{<:Machine{<:Deterministic}}, em::StackedMLEmulator, X_new::AbstractMatrix)
    preds = map(em.models) do m
        MMI.predict(m, MMI.table(transpose(X_new)))
    end
    # N x d, where d is the number of transformed output dimensions
    Z = reduce(hcat, preds)
    Y = inverse_transform_target(em.transform, Z)
    return Y
end

function _predict(::Type{<:Machine{<:Probabilistic}}, em::StackedMLEmulator, X_new::AbstractMatrix)
    # note that, for probabilistic predictors, predict returns distributions
    preds = map(em.models) do m
        MMI.predict(m, MMI.table(X_new))
    end
    @assert eltype(preds) <: AbstractVector{<:Normal} "Currently only normally distributed predictands are supported (got $(eltype(preds)))"
    # N x d, where d is the number of transformed output dimensions
    dists = reduce(hcat, preds)
    μ_pred = mean.(dists)
    σ²_pred = var.(dists)
    V = em.transform.truncated_svd.V
    sqrt_S = Diagonal(sqrt.(em.transform.truncated_svd.S))
    μs = map(eachrow(μ_pred)) do μ
        inverse_transform_target(em.transform, reshape(μ, 1, :))[1,:]
    end
    Σs = map(eachrow(σ²_pred)) do σ²
        Symmetric(V*sqrt_S*Diagonal(σ²)*sqrt_S*V')
    end
    return map(MvNormal, μs, Σs)
end
