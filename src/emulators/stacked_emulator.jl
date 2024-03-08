"""
    StackedMLEmulator{TT,TM} <: Emulator

Data structure consisting of some training data for a model emulator, a tranform operation,
and a tuple of univariate regressors which are applied to the transformed data.
"""
mutable struct StackedMLEmulator{TT,TM} <: Emulator
    data::EmulatorData
    transform::TT
    models::TM
end

function StackedMLEmulator(model::MMI.Model, data::EmulatorData, transform=NoTransform())
    X_tab = MMI.table(data.X)
    Yt = transform_target(transform, data.Y)
    machines = map(1:size(Yt,2)) do i
        yt_i = Yt[:,i]
        machine(model, X_tab, yt_i)
    end
    return StackedMLEmulator(data, transform, machines)
end

function MLJBase.fit!(em::StackedMLEmulator; mapper=map, kwargs...)
    fitted_models = mapper(_fit!, em.models, repeat([kwargs], length(em.models)))
    return StackedMLEmulator(em.data, em.transform, fitted_models)
end

MMI.predict(em::StackedMLEmulator, X_new::AbstractMatrix) = _predict(eltype(em.models), em, X_new)

_fit!(m::MLJBase.Machine, kwargs) = MLJBase.fit!(m; kwargs...)

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
    μs = map(eachrow(μ_pred)) do μ
        inverse_transform_target(em.transform, reshape(μ, 1, :))[1,:]
    end
    Σs = map(eachrow(σ²_pred)) do σ²
        Diagonal(σ²) + 1e-6*I
    end
    return map(MvNormal, μs, Σs)
end

function _predict(::Type{<:Machine{<:Probabilistic}}, em::StackedMLEmulator{DecorrelatedTarget}, X_new::AbstractMatrix)
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
        Symmetric(V*sqrt_S*Diagonal(σ²)*sqrt_S*V') + 1e-6*I
    end
    return map(MvNormal, μs, Σs)
end
