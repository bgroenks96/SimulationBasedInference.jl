"""
    StackedRegressors{TM}

Simple representation of a set of stacked, univaraite regressors for multi-target regression problems.
"""
mutable struct StackedRegressors{TM}
    models::Vector{TM}
end

stacked(model, N::Int) = StackedRegressors(repeat([model], N))
stacked(models...) = StackedRegressors(collect(models))
stacked(models::AbstractVector) = StackedRegressors(models)

predict(em::StackedRegressors, X::AbstractVector) = predict(em, reshape(X, :, 1))

function predict(em::StackedRegressors, X::AbstractMatrix)
    preds = map(m -> reshape(predict(m, X), 1, :), em.models)
    return reduce(vcat, preds)
end

function fit!(em::StackedRegressors, X, Y; mapfunc=map, kwargs...)
    ndims = size(Y,1)
    fitted_models = mapfunc(1:ndims, em.models) do i, m
        fit!(m, X, Y[i,:]; kwargs...)
    end
    return StackedRegressors(fitted_models)
end
