abstract type DataTransform end

struct NoTransform <: DataTransform end

apply(::NoTransform, Y) = Y

apply_inverse(::NoTransform, Y) = Y

struct Centered{meanType} <: DataTransform
    mean::meanType
end

Centered(X::AbstractMatrix; dims=2) = Centered(mean(X, dims=dims))

apply(c::Centered, X) = X .- c.mean

apply_inverse(c::Centered, Z) = Z .+ c.mean

struct Standardized{meanType,scaleType} <: DataTransform
    mean::meanType
    scale::scaleType
end

Standardized(X::AbstractMatrix; dims=2) = Standardized(mean(X, dims=dims), std(X, dims=dims))

apply(s::Standardized, X) = (X .- s.mean) ./ s.scale

apply_inverse(s::Standardized, Z) = Z.*s.scale .+ s.mean

"""
    Decorrelated{bijType} <: DataTransform

Decorrelation transform applied to a multivariate output (target) space.
"""
struct Decorrelated{bijType} <: DataTransform
    mean::AbstractVector
    cov::AbstractMatrix
    truncated_svd::SVD
    bijection::bijType
end

function Decorrelated(X::AbstractMatrix; frac=1.0, bijection=identity)
    bij = bijection != identity ? Base.Fix1(broadcast, bijection) : identity
    Y = bij.(X)
    Y_mean = mean(Y, dims=2)
    Y_c = Y .- Y_mean
    C_Y = cov(Y_c, dims=2)
    C_svd = svd(C_Y)
    rank = findfirst(>=(frac), cumsum(C_svd.S ./ sum(C_svd.S)))
    rank = isnothing(rank) ? length(C_svd.S) : rank
    Vt_r = C_svd.Vt[1:rank,:]
    S_r = C_svd.S[1:rank]
    truncated_svd = SVD(C_svd.U, S_r, Vt_r)
    return Decorrelated(dropdims(Y_mean, dims=2), C_Y, truncated_svd, bij)
end

function apply(d::Decorrelated, X::AbstractMatrix)
    X_centered = d.bijection.(X) .- d.mean
    S_r = d.truncated_svd.S
    Z = inv(Diagonal(sqrt.(S_r)))*d.truncated_svd.Vt*X_centered
    return Z
end

function apply_inverse(d::Decorrelated, Z::AbstractMatrix; apply_bijection=true)
    S_r = d.truncated_svd.S
    X_c = d.truncated_svd.V*Diagonal(sqrt.(S_r))*Z
    X = X_c .+ d.mean
    if apply_bijection
        f = inverse(d.bijection)
        return f.(X)
    else
        return X
    end
end

# inverse decorrelation transform for Gaussian predictions
function apply_inverse(d::Decorrelated, dists::AbstractMatrix{<:Normal}; apply_bijection=true)
    μ_pred = mean.(dists)
    σ²_pred = var.(dists)
    V = d.truncated_svd.V
    sqrt_S = Diagonal(sqrt.(d.truncated_svd.S))
    μs = map(eachcol(μ_pred)) do μ
        apply_inverse(d, reshape(μ, :, 1), apply_bijection=false)[:,1]
    end
    Σs = map(eachcol(σ²_pred)) do σ²
        Symmetric(V*sqrt_S*Diagonal(σ²)*sqrt_S*V') + 1e-6*I
    end
    f = inverse(d.bijection)
    if f === identity || !apply_bijection
        return map(MvNormal, μs, Σs)
    else
        return map((μ, Σ) -> Bijectors.TransformedDistribution(MvNormal(μ, Σ), f), μs, Σs)
    end
end

# generic affine inverse transform for Gaussian probabilistic models;
# applies the inverse to the mean of the Gaussian, which may not be valid for all
# types of transforms!
function apply_inverse(d::DataTransform, dists::AbstractMatrix{<:Normal}; kwargs...)
    μ_pred = mean.(dists)
    σ²_pred = var.(dists)
    μs = map(eachcol(μ_pred)) do μ
        apply_inverse(em.transform, reshape(μ, :, 1); kwargs...)[:,1]
    end
    Σs = map(eachcol(σ²_pred)) do σ²
        Diagonal(σ²) + 1e-6*I
    end
    return map(MvNormal, μs, Σs)
end
