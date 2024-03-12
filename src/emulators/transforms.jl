abstract type EmulatorDataTransform end

struct NoTransform <: EmulatorDataTransform end

transform_target(::NoTransform, Y) = Y

inverse_transform_target(::NoTransform, Y) = Y

struct CenteredTarget <: EmulatorDataTransform
    Y_mean
end

CenteredTarget(data::EmulatorData) = CenteredTarget(mean(data.Y, dims=1))

transform_target(c::CenteredTarget, Y) = Y .- c.Y_mean

inverse_transform_target(c::CenteredTarget, Yt) = Yt .+ c.Y_mean

"""
    DecorrelatedTarget <: EmulatorDataTransform

Decorrelation transform applied to a multivariate output (target) space.
"""
struct DecorrelatedTarget <: EmulatorDataTransform
    mean::AbstractVector
    cov::AbstractMatrix
    truncated_svd::SVD
end

function DecorrelatedTarget(data::EmulatorData; frac=1.0)
    # transpose to get matrix as d x N
    Y = transpose(data.Y)
    Y_mean = mean(Y, dims=2)
    Y_c = Y .- Y_mean
    C_Y = cov(Y_c, dims=2)
    C_svd = svd(C_Y)
    rank = findfirst(>=(frac), cumsum(C_svd.S ./ sum(C_svd.S)))
    rank = isnothing(rank) ? length(C_svd.S) : rank
    Vt_r = C_svd.Vt[1:rank,:]
    S_r = C_svd.S[1:rank]
    truncated_svd = SVD(C_svd.U, S_r, Vt_r)
    return DecorrelatedTarget(dropdims(Y_mean, dims=2), C_Y, truncated_svd)
end

function transform_target(d::DecorrelatedTarget, Y::AbstractMatrix)
    Y_centered = Y' .- d.mean
    S_r = d.truncated_svd.S
    Z = inv(Diagonal(sqrt.(S_r)))*d.truncated_svd.Vt*Y_centered
    return Z'
end

function inverse_transform_target(d::DecorrelatedTarget, Z::AbstractMatrix)
    S_r = d.truncated_svd.S
    Y_c = d.truncated_svd.V*Diagonal(sqrt.(S_r))*Z'
    Y = Y_c .+ d.mean
    return Y'
end
